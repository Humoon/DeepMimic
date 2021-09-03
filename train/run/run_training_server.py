import horovod.tensorflow as hvd
from train.common.utils import setup_logger
from train.training_server import TrainingServer

hvd.init()

watcher_logger = setup_logger("training_watcher")


def run_server():
    try:
        s = TrainingServer(use_distribution=True)
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


run_server()
