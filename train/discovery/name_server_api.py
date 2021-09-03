import socket, random, uuid, time, zmq
from typing import List, Tuple, Dict
from sys import platform
from net.zmq_adaptor import ZmqAdaptor
from train.common.utils import setup_logger
from train.common.config import is_development
from train.discovery.name_server import NameServer
from threading import Thread, Lock
from train.common.config import read_config


def host_ip() -> str:
    if is_development():
        if platform == "win32":
            # windows 开发环境
            return "127.0.0.1"
        else:
            # linux or docker 开发环境
            return "0.0.0.0"
    else:
        return socket.gethostbyname(socket.getfqdn(socket.gethostname()))


# 随机选取本地可用端口
def random_port() -> int:
    while True:
        port = random.randint(10000, 14000)
        if is_local_open(port):
            return port


# 测试本定端口是否被占用
def is_local_open(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host_ip(), port))
        return True
    except:
        return False
    finally:
        sock.close()


class NameServerAPI:
    """ 服务注册与发现调用接口
    """

    def __init__(self) -> None:
        self.logger = setup_logger("name_server_api")

        self._net = ZmqAdaptor(logger=self.logger)

        self.config = read_config("config/name.yaml")
        self._net.start({"mode": "req", "host": self.config["host"], "port": self.config["port"]})

        self.uid = str(uuid.uuid1())

        self.register_done = False

        self._req_lock = Lock()

        self._myself_info = None

    def register(self, address=None, port=None, rtype=None, extra=None) -> Tuple[str, int]:
        self._req_lock.acquire()
        api = NameServer.Req.REGISTER

        if address is None:
            address = host_ip()
        if port is None:
            port = random_port()

        self.logger.info("rigister %s:%d" % (address, port))

        req = {"address": address, "port": port, "uuid": self.uid, "rtype": rtype, "extra": extra}
        self._myself_info = req
        self._net.send_request_api(api.value, req)

        msg = self._net.receive_response_pyobj()

        if msg.get("res") != NameServer.Res.OK.value:
            assert ("name service failed {}".format(msg.get("res")))

        self.register_done = True
        self._req_lock.release()
        self._start_keep_alive_thread()
        return address, port

    def register_gpu(self, address=None, gpu_index=None) -> Tuple[str, int]:
        self._req_lock.acquire()
        api = NameServer.Req.REGISTER_GPU

        if address is None:
            address = host_ip()

        self.logger.info("rigister %s" % (address))
        req = {"address": address, "uuid": self.uid}

        # gpu_index 为 None 的时候，返回指定的一个空闲 GPU 资源
        # TrainServer 的 GPU 可能是 mpi local_rank 指定的，这个时候传入 gpu_index
        if gpu_index is not None:
            req["gpu_index"] = gpu_index

        self._net.send_request_api(api.value, req)
        msg = self._net.receive_response_pyobj()
        gpu_index = msg["data"]
        self._req_lock.release()

        return address, gpu_index

    def _start_keep_alive_thread(self):
        self._keep_alive_time = 10
        self._keep_alive_thread = Thread(target=self.keep_alive)
        self._keep_alive_thread.daemon = True
        self._keep_alive_thread.start()

    def keep_alive(self):
        while True:
            # 现阶段仅服务验活
            if self.register_done is True:
                self._req_lock.acquire()
                try:
                    api = NameServer.Req.KEEP_ALIVE
                    msg = {}
                    if self._myself_info is not None:
                        msg = self._myself_info
                    self._net.send_request_api(api.value, msg)
                    self._net.receive_response_pyobj()
                    if random.randint(0, 10) == 1:
                        self.logger.debug("send keep alive done")
                except zmq.Again:
                    self.logger.warning("keep alive fail")
                self._req_lock.release()

            time.sleep(self._keep_alive_time)

    def discovery_service(self, service_api, block=False) -> Tuple[str, List[NameServer.Service]]:
        while True:
            self._req_lock.acquire()

            try:
                self._net.send_request_api(service_api.value, {})
                msg = self._net.receive_response_pyobj()
            except zmq.Again:
                self.logger.warning("zmq send discovery_service fail")
                self._net.requester.close()
                self._net.start({"mode": "req", "host": self.config["host"], "port": self.config["port"]})
                continue
            finally:
                self._req_lock.release()

            if block is False:
                return msg["res"], msg["data"]
            elif len(msg["data"]) != 0 and msg["res"] == "ok":
                return msg["res"], msg["data"]
            else:
                time.sleep(2)

    def discovery(self, block=False) -> Tuple[str, List[NameServer.Service]]:
        return self.discovery_service(service_api=NameServer.Req.DISCOVERY_ALL, block=block)

    def discovery_q_server(self, block=False) -> Tuple[str, List[NameServer.Service]]:
        return self.discovery_service(service_api=NameServer.Req.DISCOVERY_INFERENCE, block=block)

    def discovery_train_server(self, block=False) -> Tuple[str, List[NameServer.Service]]:
        return self.discovery_service(service_api=NameServer.Req.DISCOVERY_TRAIN, block=block)

    def discovery_agents(self, block=False) -> Tuple[str, List[NameServer.Service]]:
        return self.discovery_service(service_api=NameServer.Req.DISCOVERY_AGENTS, block=block)

    def discovery_log_server(self, block=False) -> Tuple[str, List[NameServer.Service]]:
        return self.discovery_service(service_api=NameServer.Req.DISCOVERY_LOG_SERVER, block=block)

    def discovery_pub_model_server(self, block=False) -> Tuple[str, List[NameServer.Service]]:
        return self.discovery_service(service_api=NameServer.Req.DISCOVERY_PUB_MODEL_SERVER, block=block)

    def discovery_evaluation_server(self, block=False) -> Tuple[str, List[NameServer.Service]]:
        return self.discovery_service(service_api=NameServer.Req.DISCOVERY_EVALUATION_SERVER, block=block)

    def discovery_all_gpu(self) -> Tuple[str, Dict]:
        service_api = NameServer.Req.DISCOVERY_ALL_GPU
        req = {"address": host_ip()}

        self._req_lock.acquire()
        self._net.send_request_api(service_api.value, req)
        msg = self._net.receive_response_pyobj()
        self._req_lock.release()
        return msg["res"], msg["data"]

    def discovery_available_gpu(self) -> Tuple[str, int]:
        service_api = NameServer.Req.DISCOVERY_AVAILABLE_GPU
        req = {"address": host_ip()}

        self._req_lock.acquire()
        self._net.send_request_api(service_api.value, req)
        msg = self._net.receive_response_pyobj()
        self._req_lock.release()
        return msg["res"], msg["data"]

    def query_task(self):
        # 访问任务，返回 Train Server、Inference Server、LogServer 地址结果
        pass
