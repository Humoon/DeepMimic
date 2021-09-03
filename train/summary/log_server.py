import os, zmq, time, pickle, datetime
from train.common.utils import setup_logger
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from train.summary.summary_util import SummaryLog
from net.zmq_adaptor import ZmqAdaptor
from train.common.config import read_config, is_development
from train.discovery.name_server_api import NameServerAPI


class LogServer:
    """
    监控指标分五部分：

    name_server
    agent   （actor的信息）
    inf_server
    training_server
    model   （模型训练返回的信息）
    evaluate_server

    根据 log server 接收数据中的 msg_type 信息来分别写入 tensorboard 中。
    """

    def __init__(self):
        if not os.path.exists("./summary_log/"):
            os.makedirs("./summary_log/", exist_ok=True)
        self.logger = setup_logger("./log_server.log")
        self.zmq_adaptor = ZmqAdaptor(logger=self.logger)

        ns_api = NameServerAPI()
        _address, port = ns_api.register(rtype="log_server", extra={"data_type": "log_server"})
        self.zmq_adaptor.start({"mode": "pull", "host": "*", "port": port})

        self.game_mode = {
            0: "mode_Normal",
            1: "mode_KickOff",
            2: "mode_GoalKick",
            3: "mode_FreeKick",
            4: "mode_Corner",
            5: "mode_ThrowIn",
            6: "mode_Penalty"
        }
        self.actions = {
            0: "action_idle",
            1: "action_left",
            2: "action_top_left",
            3: "action_top",
            4: "action_top_right",
            5: "action_right",
            6: "action_bottom_right",
            7: "action_bottom",
            8: "action_bottom_left",
            9: "action_long_pass",
            10: "action_high_pass",
            11: "action_short_pass",
            12: "action_shot",
            13: "action_sprint",
            14: "action_release_direction",
            15: "action_release_sprint",
            16: "action_sliding",
            17: "action_dribble",
            18: "action_release_dribble"
        }

        self.raw_data_list = []
        self.next_print_time = 0

        param = read_config("config/ppo_train.yaml")
        env_name = param["env_name"]
        exp_name = param["exp_name"]
        time_now = datetime.datetime.now().strftime("%m%d%H%M%S")
        self.summary_writer = tf.summary.FileWriter(f"./summary_log/log_{time_now}_{env_name}_{exp_name}")
        self.summary_logger = SummaryLog(self.summary_writer)

        self.worker_ip_dict = {}
        self.inf_server_ip_dict = {}
        self.next_check_distinct_worker_ip_time = time.time() + 60 * 3
        self.next_check_distinct_inf_server_ip_time = time.time() + 60 * 3

        # tags
        self.name_server_tags_config = read_config("config/moni/name_server.yaml")
        self.name_server_tags = []
        self.agent_tags_config = read_config("config/moni/agent.yaml")
        self.agent_tags = []
        self.inf_server_tags_config = read_config("config/moni/inf_server.yaml")
        self.inf_server_tags = []
        self.train_tags_config = read_config("config/moni/training_server.yaml")
        self.train_tags = []
        self.model_tags_config = read_config("config/moni/model.yaml")
        self.model_tags = []
        self.evaluate_server_tags_config = read_config("config/moni/evaluate_server.yaml")
        self.evaluate_server_tags = []

    def summary_definition(self):

        # name_server
        for tag_func, tags in self.name_server_tags_config.items():
            output_freq = 10
            for tag in tags:
                self.summary_logger.add_tag("name_server/{0}_{1}".format(tag, tag_func), output_freq, tag_func)
            self.name_server_tags.extend(tags)

        # agent
        for tag_func, tags in self.agent_tags_config.items():
            for tag in tags:
                # episode end info
                if tag in [
                        "net_goals", "left_score", "right_score", "ball_owned_steps", "middle_ball_steps", "enemy_ball_steps",
                        "episode_step", "agent_sample_ratio", "goal_rate", "pass_rate", "yellow_card", "steal_ball",
                        "loose_ball", "ep_reward", "offside"
                ] or "action" in tag or "mode" in tag:
                    output_freq = 500
                    if is_development():
                        output_freq = 5
                else:
                    output_freq = 10000000
                    if is_development():
                        output_freq = 5000
                self.summary_logger.add_tag("agent/{0}_{1}".format(tag, tag_func), output_freq, tag_func)
            self.agent_tags.extend(tags)

        # inf_server
        for tag_func, tags in self.inf_server_tags_config.items():
            for tag in tags:
                if tag in ["inf_server_count"]:
                    output_freq = 100
                else:
                    output_freq = 10000
                if is_development():
                    output_freq = 5
                self.summary_logger.add_tag("inf_server/{0}_{1}".format(tag, tag_func), output_freq, tag_func)
            self.inf_server_tags.extend(tags)

        # training_server
        for tag_func, tags in self.train_tags_config.items():
            for tag in tags:
                output_freq = 100
                if is_development():
                    output_freq = 5
                self.summary_logger.add_tag("training_server/{0}_{1}".format(tag, tag_func), output_freq, tag_func)
            self.train_tags.extend(tags)

        # model
        for tag_func, tags in self.model_tags_config.items():
            for tag in tags:
                output_freq = 50
                self.summary_logger.add_tag("model/{0}_{1}".format(tag, tag_func), output_freq, tag_func)
            self.model_tags.extend(tags)

        # evaluate_server
        for tag_func, tags in self.evaluate_server_tags_config.items():
            for tag in tags:
                if tag == "net_goals":
                    output_freq = 50
                else:
                    output_freq = 10
                self.summary_logger.add_tag("evaluate_server/{0}_{1}".format(tag, tag_func), output_freq, tag_func)
            self.evaluate_server_tags.extend(tags)

        self.logger.info("name_server_tags, {}", self.name_server_tags)
        self.logger.info("agent_tags, {}", self.agent_tags)
        self.logger.info("inf_server_tags, {}", self.inf_server_tags)
        self.logger.info("trains_tags, {}", self.train_tags)
        self.logger.info("model_tags, {}", self.model_tags)
        self.logger.info("evaluate_server_tags, {}", self.evaluate_server_tags)

    def log_detail(self, data):

        if data.get("msg_type") == "name_server":
            for tag_func, tags in self.name_server_tags_config.items():
                for data_key in data.keys():
                    if data_key in tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("name_server/{0}_{1}".format(tag, tag_func), values)

        if data.get("msg_type") == "agent":
            self.worker_ip_dict[data["ip"]] = 0
            if "env_steps" in data.keys():
                self.summary_logger.total_env_steps += sum(data["env_steps"])
            self.summary_logger.add_summary("agent/env_steps_total", self.summary_logger.total_env_steps)
            if "actions" in data.keys():
                for i in range(19):
                    self.summary_logger.list_add_summary("agent/{0}_avg".format(self.actions[i]), [data["actions"][0][i]])
            if "game_mode_count" in data.keys():
                for i in range(7):
                    self.summary_logger.list_add_summary("agent/{0}_avg".format(self.game_mode[i]),
                                                         [data["game_mode_count"][0][i]])

            # check if all docker alive every 5 min
            if time.time() > self.next_check_distinct_worker_ip_time:
                self.next_check_distinct_worker_ip_time = time.time() + 60 * 5
                self.summary_logger.add_summary("agent/docker_num_5m_avg", len(self.worker_ip_dict))
                self.worker_ip_dict = {}

            for tag_func, tags in self.agent_tags_config.items():
                for data_key in data.keys():
                    if data_key in tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("agent/{0}_{1}".format(tag, tag_func), values)

        if data.get("msg_type") == "inf_server":
            self.inf_server_ip_dict[data["ip"]] = 0
            if time.time() > self.next_check_distinct_inf_server_ip_time:
                self.next_check_distinct_inf_server_ip_time = time.time() + 60 * 5
                self.summary_logger.add_summary("inf_server/inf_server_count_avg", len(self.inf_server_ip_dict))
                self.inf_server_ip_dict = {}
            for tag_func, tags in self.inf_server_tags_config.items():
                for data_key in data.keys():
                    if data_key in tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("inf_server/{0}_{1}".format(tag, tag_func), values)

        if data.get("msg_type") == "training_server":
            if "training_steps_total" in data.keys():
                self.summary_logger.add_summary("training_server/training_steps_total", max(data.pop("training_steps_total")))
            for tag_func, tags in self.train_tags_config.items():
                for data_key in data.keys():
                    if data_key in tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("training_server/{0}_{1}".format(tag, tag_func), values)

        if data.get("msg_type") == "model":
            for tag_func, tags in self.model_tags_config.items():
                for data_key in data.keys():
                    if data_key in tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("model/{0}_{1}".format(tag, tag_func), values)

        if data.get("msg_type") == "evaluate_server":
            for tag_func, tags in self.evaluate_server_tags_config.items():
                for data_key in data.keys():
                    if data_key in tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("evaluate_server/{0}_{1}".format(tag, tag_func), values)

        if data.get("msg_type") == "replay":
            self.logger.info(data["dump_name"])
            with open("./log/{0}-{1}_".format(data["score"][0], data["score"][1]) + data["dump_name"], "wb") as pickle_out:
                for one_step_replay in data["replay"]:
                    pickle.dump(one_step_replay, pickle_out)

    def run(self):
        self.summary_definition()
        self.logger.info('log begin')
        while True:
            if time.time() > self.next_print_time:
                self.summary_writer.flush()
                self.next_print_time = time.time() + 10
            # self.summary_logger.generate_time_data_output(self.logger)
            socks = dict(self.zmq_adaptor.poller.poll(timeout=100))

            # self.logger.info("log server rec")
            if self.zmq_adaptor.receiver in socks and socks[self.zmq_adaptor.receiver] == zmq.POLLIN:
                while True:
                    try:
                        data = self.zmq_adaptor.receiver.recv(zmq.NOBLOCK)
                        self.raw_data_list.append(data)
                    except zmq.ZMQError as e:
                        if type(e) != zmq.error.Again:
                            self.logger.warn("recv zmq {}".format(e))
                        break
            for raw_data in self.raw_data_list:
                # self.log_basic_info.info("receive")
                data = pickle.loads(raw_data)
                if type(data) == dict:
                    data = [data]
                for log in data:
                    if "msg_type" in log and log["msg_type"] == "error":
                        if "error_msg" in log:
                            self.logger.error(log["error"])
                    # self.logger.info("logger receive data msg_type: {}".format(log["msg_type"]))
                    self.log_detail(log)

            self.raw_data_list = []
            time.sleep(1)
