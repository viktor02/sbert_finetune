import html
import json
import logging
import os
import sqlite3
import sys
import time
from typing import List, Optional

import feedparser
import requests
import telebot
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
RSS_URL = "https://habr.com/ru/rss/articles/?fl=ru"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 300))
DB_PATH = os.getenv("DB_PATH", "/app/data/bot.db")

# Validate critical config
if not BOT_TOKEN or not CHANNEL_ID:
    # We log to stderr so it shows up as an error in container logs
    print("CRITICAL: BOT_TOKEN and CHANNEL_ID must be set.", file=sys.stderr)
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HabrBot")

# --- Database (SQLite) ---
def init_db():
    """Initialize SQLite database for tracking seen articles."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS seen_articles (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            # Cleanup old entries to keep DB small (keep last 1000)
            conn.execute(
                """DELETE FROM seen_articles WHERE id NOT IN (
                    SELECT id FROM seen_articles ORDER BY timestamp DESC LIMIT 1000
                )"""
            )
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

def is_article_seen(article_id: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT 1 FROM seen_articles WHERE id = ?", (article_id,))
            return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"DB Read Error: {e}")
        return False

def mark_article_seen(article_id: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT OR IGNORE INTO seen_articles (id) VALUES (?)", (article_id,))
    except sqlite3.Error as e:
        logger.error(f"DB Write Error: {e}")

# --- Network Logic ---
def get_requests_session():
    """Creates a session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def check_api(article_url: str) -> Optional[dict]:
    """Queries the classification API."""
    payload = {"url": article_url}
    headers = {"Content-Type": "application/json"}
    session = get_requests_session()

    try:
        response = session.post(API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API Request failed for {article_url}: {e}")
        return None

# --- Bot Logic ---
bot = telebot.TeleBot(BOT_TOKEN)

def send_telegram_notification(entry, api_response):
    verdict = api_response.get("verdict", "N/A")
    reason = api_response.get("reason", "No reason provided")
    avg_score = api_response.get("avg_ai_score", 0)

    # Escape HTML special characters to prevent broken tags
    safe_title = html.escape(entry.title)
    safe_reason = html.escape(str(reason))
    safe_verdict = html.escape(str(verdict))

    icon = "🤖" if "AI" in verdict else "✍️"

    message_text = (
        f"{icon} <b>Новая статья на Хабре</b>\n\n"
        f"<a href='{entry.link}'>{safe_title}</a>\n\n"
        f"<b>Verdict:</b> {safe_verdict}\n"
        f"<b>Score:</b> {avg_score:.2f}\n"
        f"<i>{safe_reason}</i>"
    )

    try:
        bot.send_message(
            CHANNEL_ID,
            message_text,
            parse_mode="HTML",
            disable_web_page_preview=False,
        )
        logger.info(f"Sent notification for: {entry.title}")
        return True
    except telebot.apihelper.ApiTelegramException as e:
        logger.error(f"Telegram API Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending message: {e}")
        return False

def process_rss():
    logger.info("Checking RSS feed...")

    try:
        feed = feedparser.parse(RSS_URL)
    except Exception as e:
        logger.error(f"Failed to fetch RSS feed: {e}")
        return

    if feed.bozo:
        logger.warning(f"RSS Parse Warning (Bozo): {feed.bozo_exception}")

    if not feed.entries:
        logger.info("No entries found in RSS.")
        return

    # Process from oldest to newest to maintain timeline
    for entry in reversed(feed.entries):
        article_id = entry.id

        if is_article_seen(article_id):
            continue

        logger.info(f"Processing new article: {entry.title}")

        # Check API
        api_response = check_api(entry.link)

        if not api_response:
            logger.warning(f"Skipping {entry.title} due to API failure.")
            # We do NOT mark as seen, so we try again next loop
            continue

        # Send Message
        success = send_telegram_notification(entry, api_response)

        if success:
            mark_article_seen(article_id)
            # Rate limit to avoid hitting Telegram limits
            time.sleep(2)

def main():
    logger.info("Starting HabrFilter Bot...")

    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()

    logger.info(f"Monitoring {RSS_URL}")
    logger.info(f"API Target: {API_URL}")

    while True:
        try:
            process_rss()
        except Exception as e:
            logger.exception("Critical error in main loop")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
