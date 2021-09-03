import time, zmq, pickle, copy, pprint, random, os, signal
import json
import numpy as np
from enum import Enum

from net.zmq_adaptor import ZmqAdaptor
from net.wx import send_wx
from train.common.utils import create_path, setup_logger
from train.common.config import is_development, read_config
from train.discovery.name_server_api import NameServerAPI


class EvaluationServer:
    """ 模型池服务

    训练使用： self.play_mode="self-play" or self.play_mode="build-in-bot"
    训练过程中默认使用固定分数的竞技场作为评估。

    竞技场使用：self.play_mode="arena"
    """

    class Req(Enum):
        """ 枚举服务请求类型
        """
        FINISH_MATCH = "finish_match"
        GET_POOL_MODEL = "get_pool_model"
        GET_LATEST_MODEL = "get_latest_model"
        GET_ARENA_MODEL = "get_arena_model"
        QUERY_TASK = "query_task"
        ALL = "all"

    class Res(Enum):
        """ 枚举返回相应类型
        """
        OK = "ok"
        INVALID_PARAM = "invalid_param"
        INVALID_API = "invalid_api"
        NOT_FOUND_MODEL = "not_found_model"

    def __init__(self):
        self.ns = NameServerAPI()
        self.logger = setup_logger("evaluation_server")
        self._net = ZmqAdaptor(logger=self.logger)

        conf = read_config("config/evaluation.yaml")
        # play_mode set in config/evaluation.yaml: build-in-bot, self-play, arena (init)
        self.play_mode = conf["play_mode"]

        address, port = self.ns.register(rtype="evaluation_server", extra={"data_type": "evaluation", "zmq_mode": "rep"})
        self._net.start({"mode": "rep", "host": "*", "port": port})

        self.latest_model_dict = None

        self.model_pool = {}
        self.model_result_table = {
            "build-in-bot": [],
            "training": [],
        }
        self.update_model_time = time.time() + 60 * 20

        self.arena_pool = {"build-in-bot": None}
        self.arena_info = None

        self.elo = {"build-in-bot": 1500.0, "training": 1500.0}
        self.elo_new_n = 50
        self.elo_k = 16 # 新模型进行评估的时候应增大到 2k，elo_new_n 场之后减小至原来 k，如果模型池足够大，高端排名应该缩小 k

        self.inference_servers = {} # server_uuid -> model_time

        # sub model
        # net clinet to publish model server
        if self.play_mode != 'arena':
            _res, pub_serivces = self.ns.discovery_pub_model_server(block=True)
            self._net.start({"mode": "sub", "host": pub_serivces[0].address, "port": pub_serivces[0].port})

        _res, log_servers = self.ns.discovery_log_server(block=True)
        self._net.start({"mode": "push", "host": log_servers[0].address, "port": log_servers[0].port, "dest": "logger"})

        self.moni_check_time = time.time() + 60
        self.send_wx_next_time = time.time() + 60 * 30
        self.next_dump_time = time.time() + 60 * 60
        self.kill_now = False

        self.warm_up_flag = False

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def run(self):
        if self.play_mode == "self-play":
            self.load_model_pool()
            self.load_arena_pool()
        if self.play_mode == "arena":
            self.load_arena_pool()

        while not self.kill_now:

            if self.play_mode != 'arena':
                socks = dict(self._net.poller.poll())
                self.receive_model(socks)
            else:
                socks = dict(self._net.poller.poll(timeout=1))

            # 最新的分数信息保存到本地
            if self.play_mode == 'arena' and time.time() > self.moni_check_time:
                with open('./arena_pool/arena_info.json', 'w') as fp:
                    json.dump(self.arena_info, fp, indent=4, sort_keys=True)

            self.check_moni()

            # 发送 elo 到企业微信群
            self.check_wx()
            # 备份模型
            if self.play_mode == "self-play":
                self.dump_model_pool()

            # TODO 计算、量化纳什均衡点
            if self._net.has_rep_data(socks):
                api, msg = self._net.receive_api_request()

                start_time = time.time()
                if api == EvaluationServer.Req.FINISH_MATCH.value:
                    self.finish_match(msg)
                elif api == EvaluationServer.Req.QUERY_TASK.value:
                    self.query_task()
                elif api == EvaluationServer.Req.GET_POOL_MODEL.value:
                    self.get_pool_model()
                elif api == EvaluationServer.Req.GET_LATEST_MODEL.value:
                    self.get_latest_model()
                elif api == EvaluationServer.Req.GET_ARENA_MODEL.value:
                    self.get_arena_model()
                else:
                    self._net.send_response_api({"res": EvaluationServer.Res.INVALID_API.value})
                end_time = time.time()
                used_time = (end_time - start_time) * 1000
                # self.logger.info("handle api {0} {1} ms".format(api, used_time))

        self.logger.info("dump model pool when exit")
        self._dump_model_pool()
        self.logger.info("exit EvaluationServer gracefully")

    # 结束比赛，结算elo
    def finish_match(self, msg):
        if msg["match_type"] == "self-pool":
            left_model_name = msg["left"]
            right_model_name = msg["right"]
            match_result = msg["result"]

            if match_result == 1:
                self.model_result_table[left_model_name].append("win")
                self.model_result_table[right_model_name].append("lose")
            elif match_result == 0:
                self.model_result_table[right_model_name].append("win")
                self.model_result_table[left_model_name].append("lose")
            elif match_result == 0.5:
                self.model_result_table[right_model_name].append("draw")
                self.model_result_table[left_model_name].append("draw")

            if len(self.model_result_table[left_model_name]) > 100:
                self.model_result_table[left_model_name].pop(0)

            if len(self.model_result_table[right_model_name]) > 100:
                self.model_result_table[right_model_name].pop(0)

            if left_model_name in self.elo and right_model_name in self.elo and left_model_name != right_model_name:
                left_elo, right_elo = self._cal_elo(left_model_name, right_model_name, match_result)
                # vs self-pool only update pool model's elo
                # self.elo[left_model_name] = left_elo
                self.elo[right_model_name] = right_elo
                self._net.send_response_api({"res": EvaluationServer.Res.OK.value})
            else:
                self._net.send_response_api({"res": EvaluationServer.Res.INVALID_PARAM.value})
        elif msg["match_type"] == "evaluation":
            left_model_name = msg["left"]
            right_model_name = msg["right"]
            match_result = msg["result"]
            left_elo, right_elo = self._cal_elo(left_model_name, right_model_name, match_result)
            # vs arena model only update training model's elo
            self.elo[left_model_name] = left_elo
            # self.elo[right_model_name] = right_elo
            # self.arena_info[left_model_name]["elo"] = left_elo
            # self.arena_info[left_model_name]["match_times"] += 1
            # self.arena_info[right_model_name]["elo"] = right_elo
            # self.arena_info[right_model_name]["match_times"] += 1
            self._net.send_response_api({"res": EvaluationServer.Res.OK.value})
        elif msg["match_type"] == "arena":
            left_model_name = msg["left"]
            right_model_name = msg["right"]
            match_result = msg["result"]
            left_elo, right_elo = self._cal_elo(left_model_name, right_model_name, match_result)
            self.elo[left_model_name] = left_elo
            self.elo[right_model_name] = right_elo
            self.arena_info[left_model_name]["elo"] = left_elo
            self.arena_info[left_model_name]["match_times"] += 1
            self.arena_info[right_model_name]["elo"] = right_elo
            self.arena_info[right_model_name]["match_times"] += 1
            self._net.send_response_api({"res": EvaluationServer.Res.OK.value})
        else:
            self._net.send_response_api({"res": EvaluationServer.Res.INVALID_PARAM.value})

    def get_pool_model(self):
        model_names = list(self.model_pool.keys())
        elo = np.array(list(self.elo[name] for name in model_names)) # elo 越高，采样率越高
        elo_prob = elo / elo.sum(axis=0)
        model_name = np.random.choice(model_names, p=elo_prob)

        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "model_dict": self.model_pool[model_name],
            "model_name": model_name
        })

    def get_two_arena_models(self):
        model_names = list(self.arena_pool.keys())
        left_model_name, right_model_name = np.random.choice(model_names, size=2, replace=False)
        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "play_mode": self.play_mode,
            "model_name": [left_model_name, right_model_name],
            "model_config": [
                self.arena_info[left_model_name]["model_config"], self.arena_info[right_model_name]["model_config"]
            ],
            "model_dict": [self.arena_pool[left_model_name], self.arena_pool[right_model_name]]
        })

    def get_arena_model(self):
        model_names = list(self.arena_pool.keys())
        model_name = np.random.choice(model_names)
        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "model_name": model_name,
            "model_config": self.arena_info[model_name]["model_config"],
            "model_dict": self.arena_pool[model_name]
        })

    def get_latest_model(self):
        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "model_dict": self.latest_model_dict,
        })

    def query_task(self):
        play_mode = self._which_play_mode()
        model_time = None
        model = None

        if play_mode == "self-pool":
            model_names = list(self.model_pool.keys())
            elo = np.array(list(self.elo[name] for name in model_names)) # elo 越高，采样率越高
            elo_prob = elo / elo.sum(axis=0)
            model_time = np.random.choice(model_names, p=elo_prob)
            model = self.model_pool[model_time]
        elif play_mode == "self-play":
            model_time = "training"
        elif play_mode == "arena":
            return self.get_two_arena_models()
        else:
            model_time = "build-in-bot"

        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "play_mode": play_mode,
            "model_name": model_time,
            "model_config": None,
            "model_dict": model
        })

    # play_mode 切换逻辑写在这里
    def _which_play_mode(self):
        if self.play_mode == "build-in-bot":
            return self.play_mode
        if self.play_mode == "arena":
            return self.play_mode

        r = random.random()
        if r < 0.2 and len(self.model_pool) > 0:
            return "self-pool"
        else:
            return "self-play"

    # 计算更新 elo, result 胜:1 平:0.5 负: 0
    def _cal_elo(self, left_model_name: str, right_model_name: str, result: float):
        left_elo = self.elo[left_model_name]
        right_elo = self.elo[right_model_name]

        E_left = 1 / (1 + 10**((right_elo - left_elo) / 400))
        E_right = 1 - E_left

        k = self.elo_k

        # 高端局递减
        if left_elo > 2000 or right_elo > 2000:
            k /= 2

        left_elo += k * (result - E_left)
        right_elo += k * ((1 - result) - E_right)

        return left_elo, right_elo

    def receive_model(self, socks):
        model_str = None
        if self._net.subscriber in socks and socks[self._net.subscriber] == zmq.POLLIN:
            model_str = self._net.subscriber.recv()

        if model_str is None:
            return

        model_dict = pickle.loads(model_str)
        self.latest_model_dict = model_dict
        # self.logger.info("evaluation receive model: {}".format(model_dict["update_times"]))
        if self.play_mode == "self-play" and "model_time" in model_dict and self._should_update(
                model_time=model_dict["model_time"]):
            self._insert_new_model(model_time=model_dict["model_time"], model_dict=model_dict)

    def _should_update(self, model_time):
        return model_time not in self.model_pool and time.time() > self.update_model_time

    def _insert_new_model(self, model_time, model_dict):
        if "elo" in model_dict and model_dict["elo"] > 1000:
            self.elo[model_time] = model_dict["elo"]
        else:
            self.elo[model_time] = self.elo["training"]
        self.model_pool[model_time] = model_dict
        self.model_result_table[model_time] = []

        self.logger.info("add new model {}".format(model_time))
        # 实现剔除逻辑[重要]
        pop_model_times = []
        for model_time, elo in self.elo.items():
            if elo < 1000 and model_time != "build-in-bot" and model_time != "training" and 'model' not in str(model_time):
                pop_model_times.append(model_time)

        for model_time in pop_model_times:
            self.model_pool.pop(model_time)
            self.elo.pop(model_time)

        if is_development():
            self.update_model_time = time.time() + 30
        else:
            self.update_model_time = time.time() + 60 * 20

    def check_moni(self):
        if time.time() > self.moni_check_time:
            tmp_moni = copy.deepcopy(self.elo)
            model_names = list(tmp_moni.keys())

            for name in model_names:
                if name not in ["training", "build-in-bot"]:
                    continue

                tmp_moni[f"elo_{name}"] = tmp_moni.pop(name)
                all_count = len(self.model_result_table[name])
                if all_count != 0:
                    tmp_moni[f"win_prob_{name}"] = self.model_result_table[name].count("win") / all_count
                    tmp_moni[f"draw_prob_{name}"] = self.model_result_table[name].count("draw") / all_count

            tmp_moni["model_pool_length"] = len(self.model_pool)
            tmp_moni['msg_type'] = "evaluate_server"
            self._net.send_log(pickle.dumps(tmp_moni))
            self.moni_check_time = time.time() + 60

    def check_wx(self):
        if time.time() > self.send_wx_next_time:
            pp = pprint.PrettyPrinter(indent=4)
            content = "elo: "
            content += pp.pformat(self.elo)
            # content += "\nresult_table: "
            # content += pp.pformat(self.model_result_table)
            content += "\n【4vs4 master】"
            if is_development():
                content += "\n【测试环境】"
            send_wx({"content": content})
            self.send_wx_next_time = time.time() + 60 * 30

    def load_arena_pool(self):
        if os.path.exists("./arena_pool") is False:
            return

        with open('./arena_pool/arena_info.json', 'r') as fp:
            self.arena_info = json.load(fp)

        self.elo["build-in-bot"] = self.arena_info["build-in-bot"]["elo"]

        files = os.listdir("./arena_pool")
        for file in files:
            if "model_f" in file:
                with open(f"./arena_pool/{file}", "rb") as f:
                    model_dict = pickle.loads(f.read())
                    self.arena_pool[model_dict["model_name"]] = model_dict
                    self.elo[model_dict["model_name"]] = self.arena_info[model_dict["model_name"]]["elo"]

        self.logger.info("load arena pool susses!")

    # 加载本地模型
    def load_model_pool(self):
        if os.path.exists("./log/model_pools") is False:
            return
        files = os.listdir("./log/model_pools")
        for file in files:
            if "model_f" in file:
                with open(f"./log/model_pools/{file}", "rb") as f:
                    model_dict = pickle.loads(f.read())
                    self._insert_new_model(model_time=model_dict["model_time"], model_dict=model_dict)
        self.logger.info("load model pool susses!")

    def dump_model_pool(self):
        if time.time() >= self.next_dump_time:
            self.next_dump_time = time.time() + 60 * 60
            self.logger.info("training elo: {0}".format(self.elo["training"]))
            self.logger.info("pool elo: {0}".format(self.elo.values()))
            self._dump_model_pool()

    # 备份模型
    def _dump_model_pool(self):
        self.logger.info(f"dump model pool {len(self.model_pool)}")
        create_path("./log/model_pools")
        for model_time, model_dict in self.model_pool.items():
            model_dict["elo"] = self.elo[model_time]
            file = f"model_f_{model_time}"
            with open(f"./log/model_pools/{file}", "wb") as f:
                f.write(pickle.dumps(model_dict))

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        self.logger.info(f"get exit_gracefully signal {signum} {frame}")
