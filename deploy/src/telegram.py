import json
import logging
import os
import time

import feedparser
import requests
import telebot

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
RSS_URL = "https://habr.com/ru/rss/articles/?fl=ru"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 300))

DB_FILE = os.getenv("DB_FILE", "seen_articles.json")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)


def load_seen_articles():
    """Загружает список ID уже обработанных статей."""
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_seen_articles(seen_list):
    """Сохраняет список ID обработанных статей (храним последние 100)."""
    # Оставляем только последние 100, чтобы файл не разрастался бесконечно
    trimmed_list = seen_list[-100:]
    # Гарантируем, что директория существует
    os.makedirs(os.path.dirname(os.path.abspath(DB_FILE)), exist_ok=True)

    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed_list, f)


def check_api(article_url):
    """Делает запрос к локальному API для проверки статьи."""
    payload = {"url": article_url}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе к API ({API_URL}): {e}")
        return None


def process_rss():
    """Основная функция проверки RSS и отправки сообщений."""
    logger.info("Проверка RSS ленты...")

    seen_articles = load_seen_articles()
    feed = feedparser.parse(RSS_URL)

    if feed.bozo:
        logger.error("Ошибка при парсинге RSS ленты")
        return

    # RSS обычно идет от новых к старым, нам лучше обрабатывать наоборот,
    # чтобы соблюдать хронологию при первой загрузке, но для мониторинга это не критично.
    # Проходимся по списку.

    new_articles_found = False

    # Инвертируем список, чтобы старые (новые для нас) обрабатывались первыми, если их несколько
    for entry in reversed(feed.entries):
        article_id = entry.id  # Уникальный ID статьи в RSS
        article_url = entry.link
        article_title = entry.title

        if article_id not in seen_articles:
            logger.info(f"Найдена новая статья: {article_title}")

            # 1. Делаем запрос к API
            api_response = check_api(article_url)

            if api_response:
                verdict = api_response.get("verdict", "N/A")
                reason = api_response.get("reason", "")
                avg_score = api_response.get("avg_ai_score", 0)

                # Формируем сообщение
                # Можно добавить эмодзи в зависимости от вердикта
                icon = "🤖" if "AI" in verdict else "✍️"

                message_text = (
                    f"{icon} <b>Новая статья на Хабре</b>\n\n"
                    f"<a href='{article_url}'>{article_title}</a>\n\n"
                    f"<b>Verdict:</b> {verdict}\n"
                    f"<b>Score:</b> {avg_score:.2f}\n"
                    f"<i>{reason}</i>"
                )

                try:
                    # 2. Отправляем в Telegram
                    bot.send_message(
                        CHANNEL_ID,
                        message_text,
                        parse_mode="HTML",
                        disable_web_page_preview=False,
                    )
                    logger.info(f"Сообщение отправлено для: {article_title}")

                    # 3. Добавляем в просмотренные
                    seen_articles.append(article_id)
                    new_articles_found = True

                    # Небольшая пауза, чтобы не спамить, если статей много сразу
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Ошибка при отправке в Telegram: {e}")
            else:
                logger.warning(
                    f"Не удалось получить вердикт для {article_url}, пропускаем пока."
                )

    if new_articles_found:
        save_seen_articles(seen_articles)


def main():
    logger.info("Бот запущен")
    logger.info(f"API URL: {API_URL}")
    while True:
        try:
            process_rss()
        except Exception as e:
            logger.error(f"Критическая ошибка в основном цикле: {e}")

        # Ожидание перед следующей проверкой
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
