from train.common.utils import setup_logger
from train.data_server import DataServer
from train.discovery.name_server_api import host_ip
from train.common.config import read_config
from multiprocessing import Process
from train.common.utils import setup_logger
from train.inferene_server import InferenceGpuServer

watcher_logger = setup_logger("data_server_watcher")


def run_server(index):
    try:
        s = DataServer(index)
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


current_ip = host_ip()
if current_ip == "10.70.235.21":
    num_gpus = 6
else:
    num_gpus = 8 # per machine
processes = []

for i in range(num_gpus):
    processes.append(Process(target=run_server, args=(i,)))

for t in processes:
    t.start()

for t in processes:
    t.join()
