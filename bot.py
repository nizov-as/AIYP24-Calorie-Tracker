import asyncio
import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiohttp import web
from aiogram.fsm.storage.redis import RedisStorage
import redis.asyncio as redis
from config import TOKEN
from handlers import setup_handlers
from middlewares import LoggingMiddleware

# Загружаем переменные из .env файла
load_dotenv()

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Функция для health check
async def handle_healthcheck(request):
    return web.Response(text="OK")

# Запуск HTTP-сервера в отдельном потоке
def run_web_app():
    app = web.Application()
    app.add_routes([web.get('/', handle_healthcheck)])
    web.run_app(app, port=8000, host='0.0.0.0')

# Настраиваем хранилище состояний на Redis (берём из переменных окружения)
redis_conn = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
)
storage = RedisStorage(redis=redis_conn)

# 🤖 Создаем экземпляры бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher(storage=storage)

# 🔗 Подключаем middleware и handlers
dp.message.middleware(LoggingMiddleware())
setup_handlers(dp)

async def main():
    logger.info("Бот запущен ✅")
    try:
        await dp.start_polling(bot)
    finally:
        await redis_conn.aclose()
        await storage.aclose()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.warning("🛑 Бот остановлен вручную.")