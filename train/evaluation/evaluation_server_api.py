import socket, random, uuid, time
from train.evaluation.evaluation_server import EvaluationServer
from train.discovery.name_server_api import NameServerAPI, host_ip
from typing import List, Tuple, Dict
from net.zmq_adaptor import ZmqAdaptor
from train.common.utils import setup_logger


class EvaluationServerAPI:
    """ 模型池接口 API
    """

    def __init__(self) -> None:
        self.logger = setup_logger("evaluation_server_api")
        self._net = ZmqAdaptor(logger=self.logger)
        self.ns = NameServerAPI()
        _res, evaluation_servers = self.ns.discovery_evaluation_server(block=True)

        self._net.start({"mode": "req", "host": evaluation_servers[0].address, "port": evaluation_servers[0].port})

        self.uid = str(uuid.uuid1())

    def finish_match(self, left_model_name: str, right_model_name: str, result: float, match_type: str) -> str:
        service_api = EvaluationServer.Req.FINISH_MATCH
        req = {
            "address": host_ip(),
            "uuid": self.uid,
            "left": left_model_name,
            "right": right_model_name,
            "result": result,
            "match_type": match_type
        }

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        return msg["res"]

    def query_task(self):
        service_api = EvaluationServer.Req.QUERY_TASK
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        # _res, play_mode, model_name, model_config, model_dict
        return msg["res"], msg["play_mode"], msg["model_name"], msg["model_config"], msg["model_dict"]

    def get_pool_model(self):
        service_api = EvaluationServer.Req.GET_POOL_MODEL
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        return msg["res"], msg["model_name"], msg["model_dict"]

    def get_latest_model(self):
        service_api = EvaluationServer.Req.GET_LATEST_MODEL
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        return msg["res"], msg["model_dict"]

    def get_arena_model(self):
        service_api = EvaluationServer.Req.GET_ARENA_MODEL
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        return msg["res"], msg["model_name"], msg["model_config"], msg["model_dict"]
