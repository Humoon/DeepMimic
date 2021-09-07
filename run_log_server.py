from newton.common.utils import setup_logger
from newton import LogServer, NewtonConfig

watcher_logger = setup_logger("log_server_watcher")


def run_server():
    try:
        NewtonConfig.config_path = "data/config"
        s = LogServer()
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


run_server()