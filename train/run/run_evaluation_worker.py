from train.common.utils import setup_logger
from worker.evaluation_worker import EvaluationWorker

watcher_logger = setup_logger("evaluation_server_watcher")


def run_server():
    try:
        s = EvaluationWorker()
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


run_server()
