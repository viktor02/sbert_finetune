import csv
import hashlib
import os
import re
import sqlite3

from bs4 import BeautifulSoup

# Настройки
# Используем абсолютный путь или относительный от корня проекта
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/ai_articles.db")
OUTPUT_FILE = "dataset_chunked_ai.csv"

# Настройки чанкинга
WINDOW_SIZE = 3  # Сколько предложений объединять в один кусок
STRIDE = 1  # Шаг сдвига окна (1 = максимальное перекрытие, больше данных)
MIN_WORDS = 10  # Минимальное кол-во слов в чанке, чтобы считать его полезным


def clean_html(html_content):
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Удаляем мусор: скрипты, стили, мета-данные
    for element in soup(
        ["script", "style", "meta", "noscript", "iframe", "svg", "path"]
    ):
        element.decompose()

    # 2. Удаляем блоки кода (они путают модель, если мы ищем AI-текст, а не AI-код)
    # На Хабре код часто в <pre><code ...> или просто <code>
    for element in soup(["pre", "code"]):
        # Можно заменить на токен, если хотите, чтобы модель знала, что тут был код
        # element.replace_with(" [CODE_BLOCK] ")
        element.decompose()

    # 3. Получаем текст
    text = soup.get_text(separator=" ", strip=True)

    # 4. Удаляем лишние пробелы
    text = " ".join(text.split())
    return text


def split_into_sentences(text):
    """
    Разбивает текст на предложения, сохраняя знаки препинания.
    """
    # Более сложная регулярка, чтобы не ломать "т.д.", "т.п.", "им. Ленина"
    # Но для простоты оставим надежный вариант с lookbehind
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_sliding_window(text, window_size=3, stride=1):
    """
    Создает чанки из window_size предложений, сдвигаясь на stride.
    """
    sentences = split_into_sentences(text)
    chunks = []

    if len(sentences) < window_size:
        # Если текст короче окна, берем его целиком, если он достаточно длинный
        full_text = " ".join(sentences)
        if len(full_text.split()) >= MIN_WORDS:
            return [full_text]
        return []

    # Скользящее окно
    for i in range(0, len(sentences) - window_size + 1, stride):
        group = sentences[i : i + window_size]
        chunk = " ".join(group)
        if len(chunk.split()) >= MIN_WORDS:
            chunks.append(chunk)

    return chunks


def main():
    if not os.path.exists(DB_PATH):
        print(f"Ошибка: Файл базы данных не найден по пути: {DB_PATH}")
        return
    else:
        db_file = DB_PATH

    print(f"Используем базу данных: {db_file}")

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        print("Чтение данных из базы...")
        # Убрал LIMIT для реальной работы, или поставьте обратно для теста
        cursor.execute("SELECT content FROM dataset")

        # Используем set для дедупликации на лету
        seen_hashes = set()

        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["text", "label"])

            rows = cursor.fetchall()
            total_articles = len(rows)
            total_chunks = 0
            skipped_dupes = 0

            print(f"Найдено статей: {total_articles}. Начинаю обработку...")

            for i, row in enumerate(rows):
                raw_content = row[0]
                cleaned_text = clean_html(raw_content)

                # Используем скользящее окно
                text_chunks = chunk_text_sliding_window(
                    cleaned_text, WINDOW_SIZE, STRIDE
                )

                for chunk in text_chunks:
                    # Вычисляем хэш для проверки дублей
                    chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()

                    if chunk_hash not in seen_hashes:
                        writer.writerow([chunk, 1])  # 0 - Human Label
                        seen_hashes.add(chunk_hash)
                        total_chunks += 1
                    else:
                        skipped_dupes += 1

                if (i + 1) % 50 == 0:
                    print(
                        f"Статей: {i + 1}. Чанков: {total_chunks}. Дубликатов отброшено: {skipped_dupes}"
                    )

        print(f"Готово! Из {total_articles} статей получено {total_chunks} примеров.")

    except sqlite3.Error as e:
        print(f"Ошибка SQLite: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
