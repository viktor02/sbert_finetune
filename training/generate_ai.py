import concurrent.futures
import os
import random
import sqlite3
import time

from openai import OpenAI

# --- КОНФИГУРАЦИЯ ---
DB_NAME = "data/ai_articles.db"
# Используем дешевую и быструю модель для массовости, либо умную для качества
# Рекомендую чередовать модели, если бюджет позволяет, чтобы детектор не переобучился на одну.
MODELS = [
    "anthropic/claude-3-haiku",
    "openai/o4-mini",
    "x-ai/grok-4.1-fast",
    "openai/gpt-oss-120b",
    "openai/gpt-5-nano",
]

THREADS = 5  # Количество одновременных запросов
TOTAL_ARTICLES_TO_GENERATE = 100  # Сколько всего статей хотим

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://localhost",
        "X-Title": "Dataset Generator",
    },
)

# --- ПЕРСОНЫ И СТИЛИ ---
# Разные стили заставляют модель менять структуру предложений (burstiness)
PERSONAS = [
    "Ты циничный Senior Developer, который устал от хайпа. Ты пишешь критическую статью. Используй сленг (деплой, прод, костыль).",
    "Ты восторженный Junior, который только что разобрался в сложной теме. Ты пишешь туториал 'для чайников'. Много эмоций.",
    "Ты DevOps-инженер. Пишешь сухой, четкий пост-мортем (разбор инцидента). Минимум воды, максимум фактов.",
    "Ты стартапер, который пилит свой пет-проект. Рассказываешь историю успеха и провала. Стиль повествовательный, сторителлинг.",
    "Ты эксперт по безопасности. Ты параноик. Пишешь о том, почему всё небезопасно.",
]

ANTI_AI_RULES = (
    "ВАЖНО: \n"
    "1. НЕ пиши вступлений типа 'В этой статье мы рассмотрим...'. Сразу переходи к делу.\n"
    "2. НЕ пиши заключений типа 'В заключение хочется сказать...'. Просто обрывай мысль или давай финальный совет.\n"
    "3. Избегай маркированных списков, если это не перечисление параметров. Пиши связным текстом.\n"
    "4. Используй скобки для уточнений (вот так), тире — для динамики.\n"
    "5. Допускай легкую небрежность, используй разговорные обороты.\n"
    "6. Пиши всегда на русском языке."
)

# Базовые категории для генерации тем
BASE_CATEGORIES = [
    "Python разработка",
    "DevOps и Linux",
    "Frontend (React/Vue)",
    "Data Science",
    "Информационная безопасность",
    "Карьера в IT",
    "Микроконтроллеры и DIY",
    "Базы данных",
    "Системное администрирование",
]


def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            result INTEGER DEFAULT 1,
            model TEXT,
            topic TEXT
        )
    """)
    # Миграция для старых баз
    try:
        cursor.execute("ALTER TABLE dataset ADD COLUMN topic TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


def generate_topics(count=20):
    """Генерирует список уникальных тем, чтобы не хардкодить их."""
    category = random.choice(BASE_CATEGORIES)
    prompt = (
        f"Придумай {count} кликбейтных и интересных заголовков для статей на Хабр "
        f"в категории '{category}'. Верни только список заголовков, по одному на строку."
    )
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        topics = response.choices[0].message.content.strip().split("\n")
        # Чистим от нумерации (1. Тема -> Тема)
        clean_topics = [t.split(". ", 1)[-1].strip() for t in topics if t.strip()]
        return clean_topics
    except Exception as e:
        print(f"Ошибка генерации тем: {e}")
        return []


def generate_article_task(topic):
    """Функция для выполнения в потоке."""
    model = random.choice(MODELS)
    persona = random.choice(PERSONAS)

    system_prompt = f"{persona} {ANTI_AI_RULES}"
    user_prompt = (
        f"Напиши статью на тему: '{topic}'. "
        f"Используй HTML теги (h2, p, code, b, i). "
        f"Объем: от 300 до 800 слов."
    )

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=random.uniform(
                0.7, 0.95
            ),  # Рандомная температура для разнообразия
        )
        content = response.choices[0].message.content
        duration = time.time() - start_time
        return {
            "success": True,
            "topic": topic,
            "content": content,
            "model": model,
            "duration": duration,
        }
    except Exception as e:
        return {"success": False, "topic": topic, "error": str(e)}


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ОШИБКА: OPENROUTER_API_KEY не найден.")
        return

    conn = init_db()
    print(
        f"--- Старт генерации {TOTAL_ARTICLES_TO_GENERATE} статей в {THREADS} потоков ---"
    )

    generated_count = 0

    # Пул потоков для параллельной генерации
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        while generated_count < TOTAL_ARTICLES_TO_GENERATE:
            # 1. Генерируем пачку тем
            print(">>> Генерирую новые темы...")
            topics = generate_topics(THREADS * 2)
            if not topics:
                time.sleep(5)
                continue

            # 2. Запускаем задачи на генерацию статей
            futures = {
                executor.submit(generate_article_task, topic): topic for topic in topics
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()

                if result["success"]:
                    # Сохраняем в БД (SQLite требует thread-safety, поэтому лучше делать это в главном потоке или использовать блокировки,
                    # но здесь мы просто последовательно пишем по мере готовности)
                    try:
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO dataset (content, result, model, topic) VALUES (?, ?, ?, ?)",
                            (result["content"], 1, result["model"], result["topic"]),
                        )
                        conn.commit()
                        generated_count += 1
                        print(
                            f"[{generated_count}/{TOTAL_ARTICLES_TO_GENERATE}] OK ({result['duration']:.1f}s): {result['topic'][:50]}..."
                        )
                    except Exception as db_err:
                        print(f"Ошибка записи в БД: {db_err}")
                else:
                    print(
                        f"FAIL: {result['topic'][:30]}... Ошибка: {result.get('error')}"
                    )

                if generated_count >= TOTAL_ARTICLES_TO_GENERATE:
                    break

    conn.close()
    print("--- Готово ---")


if __name__ == "__main__":
    main()
