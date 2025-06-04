import asyncio
import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.redis import Redis, RedisStorage
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

# Настраиваем хранилище состояний на Redis (берём из переменных окружения)
redis = Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
)
storage = RedisStorage(redis=redis)

# 🤖 Создаем экземпляры бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher(storage=storage)

# 🔗 Подключаем middleware и handlers
dp.message.middleware(LoggingMiddleware())
setup_handlers(dp)

async def main():
    logger.info("Бот запущен ✅")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.warning("🛑 Бот остановлен вручную.")