import os
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

# Чтение токена из переменной окружения
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise ValueError("Переменная окружения BOT_TOKEN не установлена!")

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("Переменная окружения API_KEY не установлена!")

API_KEY_TASTY = os.getenv("API_KEY_TASTY")

if not API_KEY_TASTY:
    raise ValueError("Переменная окружения API_KEY_TASTY не установлена!")

API_HOST = os.getenv("API_HOST")

if not API_HOST:
    raise ValueError("Переменная окружения API_HOST не установлена!")

API_KEY_NUTR = os.getenv("API_KEY_NUTR")

if not API_KEY_NUTR:
    raise ValueError("Переменная окружения API_KEY_NUTR не установлена!")

APP_ID = os.getenv("APP_ID")

if not APP_ID:
    raise ValueError("Переменная окружения APP_ID не установлена!")
