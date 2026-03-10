import logging
import json
import os
import sys
import traceback
import contextvars
from datetime import datetime

from logstash_async.handler import AsynchronousLogstashHandler

request_headers = contextvars.ContextVar('request_headers')

def is_remote_logger():
    env_value = os.getenv('REMOTE_LOGGER', '').lower()
    if env_value == 'true':
        logging.getLogger().info("Remote logger")
        return True
    else:
        logging.getLogger().info("Local logger")
        return False

def init_logger():
    handler = AsynchronousLogstashHandler(
        host='logstash',
        port=5022,
        database_path=None,
    ) if is_remote_logger() else logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(
        level=logging.INFO if is_remote_logger() else logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[handler]
    )

    logging.getLogger().info(f"Prod mode: {is_remote_logger()}")

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'application': 'text-to-voice',
            'level': record.levelname,
            'message': record.getMessage(),
            'logger_name': record.filename,
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
        }
        if record.exc_info:
            log_obj['exception'] = ''.join(traceback.format_exception(*record.exc_info))

        headers = request_headers.get(None)
        if isinstance(headers, dict):
            for key, value in headers.items():
                if key.startswith('custom-'):
                    log_obj[key.removeprefix('custom-')] = value

        return json.dumps(log_obj)
