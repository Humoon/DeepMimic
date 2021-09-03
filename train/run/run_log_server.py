from train.common.utils import setup_logger
from train.log_server import LogServer

watcher_logger = setup_logger("log_server_watcher")


def run_server():
    try:
        s = LogServer()
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


run_server()
