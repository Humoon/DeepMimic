from newton.common.utils import setup_logger
from newton import NameServer, NewtonConfig

watcher_logger = setup_logger("name_server_watcher")


def run_server():
    try:
        NewtonConfig.config_path = "data/config"
        s = NameServer()
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


run_server()
