from train.common.utils import setup_logger
from train.discovery.name_server import NameServer

watcher_logger = setup_logger("name_server_watcher")


def run_server():
    try:
        s = NameServer()
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


run_server()
