import logging
import sys
import json


class JSONLogger(logging.Logger):
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
    MESSAGE_FORMAT = '{"@timestamp": "%(asctime)s.%(msecs)03d", "message": %(message)s, "module": "%(module)s", "logger_name": "%(name)s", "level": "%(levelname)s"}'

    def __init__(self, name, level):
        super(JSONLogger, self).__init__(name, level)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        super()._log(level, json.dumps(msg), args, exc_info, extra, stack_info, stacklevel+2)

    @staticmethod
    def get_logger() -> 'JSONLogger':
        logger_ = JSONLogger("pacbb", logging.DEBUG)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(JSONLogger.MESSAGE_FORMAT, datefmt=JSONLogger.DATE_FORMAT))

        logger_.addHandler(console_handler)

        return logger_


logger = JSONLogger.get_logger()
