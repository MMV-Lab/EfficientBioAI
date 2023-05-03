import logging
import sys

LOGGER_NAME = "EFFICIENTBIOAI"
logger = logging.getLogger(LOGGER_NAME)
logger.propagate = False

# Log format and date format
fmt = logging.Formatter(
    fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Standard output handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(fmt)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

# File handler
log_file = "efficientbioai.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(fmt)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Set logger level and parent
logger.setLevel(logging.INFO)
logger.parent = None


def set_log_level(level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging():
    logger.handlers = []
