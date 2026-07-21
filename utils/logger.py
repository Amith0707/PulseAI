import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class SimpleLogger:
    def __init__(self, name="pulseai"):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        if self._logger.handlers:
            return

        log_file = os.path.join(LOG_DIR, f"{name}.log")

        handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        handler.suffix = "%Y-%m-%d"

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self._logger.addHandler(handler)
        self._logger.addHandler(console_handler)

    def log(self, message, level="info"):
        getattr(self._logger, level.lower())(message)


logger = SimpleLogger()