import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    LOG_FILE = os.path.join(log_dir, "app.log")
    logger = logging.getLogger("CalorieTrackerService")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger