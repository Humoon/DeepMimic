from worker.runner import Runner
from train.common.utils import setup_logger

watcher_logger = setup_logger("run_episode.log")


def run_client():
    try:
        r = Runner()
        r.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


run_client()
