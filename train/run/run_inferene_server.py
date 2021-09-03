from train.discovery.name_server_api import host_ip
from train.common.config import read_config
from multiprocessing import Process
from train.common.utils import setup_logger
from train.inferene_server import InferenceGpuServer

current_ip = host_ip()
c = read_config("config/hosts.yaml")["inference_server"]

watcher_logger = setup_logger("inference_watcher")


def run_server(device):
    try:
        s = InferenceGpuServer(device=device)
        s.run()
    except Exception as e:
        watcher_logger.exception("WHAT?!")
        watcher_logger.error(str(e))


processes = []

for server in filter(lambda x: x["ip"] == current_ip, c):
    for _ in range(server["cpu"]):
        processes.append(Process(target=run_server, args=("cpu",)))

    for _ in range(server["gpu"]):
        processes.append(Process(target=run_server, args=("gpu",)))

for t in processes:
    t.start()

for t in processes:
    t.join()
