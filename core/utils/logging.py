import logging
import sys


def get_logger() -> logging.Logger:
    date_format = "%Y-%m-%dT%H:%M:%S"
    message_format = (
        '{"@timestamp": "%(asctime)s.%(msecs)03d", "message": %(message)s, "module": "%(module)s", "logger_name": "%(name)s", "level": "%(levelname)s"}'
    )

    logger_ = logging.getLogger('emb')
    logger_.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(message_format, datefmt=date_format))

    logger_.addHandler(console_handler)

    return logger_


logger = get_logger()
