import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import zmq, pickle, time
from train.common.utils import setup_logger
from train.learner.training_set import TrainingSet
from net.zmq_adaptor import ZmqAdaptor
from train.summary.moni import Moni
from train.common.config import is_development
from train.discovery.name_server_api import NameServerAPI
from train.evaluation.evaluation_server_api import EvaluationServerAPI
from train.common.config import read_config
import importlib


class InferenceGpuServer:

    def __init__(self, device="gpu", inference_type="training"):
        self.test = is_development()
        self.device = device
        self.inference_type = inference_type # training or fixed
        param = read_config("config/ppo_train.yaml")

        self.gpu_num = 0 # config['gpu_num']
        self.player_num = param["player_num"]
        self.logger = setup_logger("./inference_gpu_server_log_{0}_{1}_log".format(self.gpu_num, self.inference_type))

        # net server
        self.zmq_adaptor = ZmqAdaptor(logger=self.logger)
        ns_api = NameServerAPI()
        address, port = ns_api.register(
            rtype="inference_server", extra={
                "data_type": self.inference_type,
                "zmq_mode": "router"
            })
        self.zmq_adaptor.start({"mode": "router", "host": "*", "port": port})
        self.ip = "%s_%d" % (address, port)
        self.logger.info(f"begin listen {self.ip}")

        # net clinet to publish model server
        if self.inference_type == "training":
            _res, pub_serivces = ns_api.discovery_pub_model_server(block=True)
            self.zmq_adaptor.start({"mode": "sub", "host": pub_serivces[0].address, "port": pub_serivces[0].port})
        else:
            # TODO
            self.ev_api = EvaluationServerAPI()
            # self.ev_api.query_inference_task()

        # net client to log server
        _res, log_services = ns_api.discovery_log_server(block=True)
        self.zmq_adaptor.start({
            "mode": "push",
            "host": log_services[0].address,
            "port": log_services[0].port,
            "dest": "logger"
        })

        # device config
        if self.device == "gpu":
            self.config = tf.ConfigProto()
        elif self.device == "cpu":
            self.config = tf.ConfigProto(device_count={"GPU": 0})

        if self.test:
            self.logger.info('debug model')
        elif self.device == "cpu":
            self.logger.info('device cpu model')
        else:
            _, self.gpu_num = ns_api.register_gpu()
            self.logger.info('device gpu model, gpu num: %d' % (self.gpu_num))
            self.config.gpu_options.visible_device_list = str(self.gpu_num)
            self.config.gpu_options.allow_growth = True

        ppo_model = importlib.import_module("model.football_models." + param['football_model'])
        self.model = ppo_model.PPOModel(param)

        self.training_set = TrainingSet(player_num=self.player_num, max_capacity=10000, training_set_type="inferene_type")
        self.moni = Moni()
        self.max_waiting_time = 0.010
        self.next_check_time = time.time()
        self.raw_data_list = []
        self.next_check_stat_time = time.time() + 60
        self.first_time_update = True
        self.receive_data_count = 0

    def receive_model(self, socks):
        model_package = None
        if self.inference_type == "training":
            if self.zmq_adaptor.subscriber in socks and socks[self.zmq_adaptor.subscriber] == zmq.POLLIN:
                model_package = self.zmq_adaptor.subscriber.recv()
            if model_package is not None:
                bt = time.time()
                self.model_time, update_times = self.model.deserializing(model_package)
                self.model.update_times = update_times
                model_delay_t = time.time() - self.model_time
                load_model_dt = time.time() - bt
                self.logger.info("load model {0},use time {1}, model delay {2}".format(update_times, load_model_dt,
                                                                                       model_delay_t))
                moni_dict = {"model_delay_time": model_delay_t, "inf_server_load_model_time": load_model_dt}
                self.moni.record(moni_dict)
        elif self.inference_type == "fixed":
            pass

    def receive_data(self, socks):
        if self.zmq_adaptor.router_receiver in socks and socks[self.zmq_adaptor.router_receiver] == zmq.POLLIN:
            tmp_list = []
            while True:
                try:
                    data = self.zmq_adaptor.router_receiver.recv_multipart(zmq.NOBLOCK)
                    tmp_list.append(data)
                except zmq.ZMQError as e:
                    break
            # self.logger.info('recieve data num {0}'.format(len(tmp_list)))
            for raw_data in tmp_list:
                self.raw_data_list.append(raw_data)
                instance = pickle.loads(raw_data[-1])
                self.training_set.append_instance(instance)
                self.receive_data_count += 1

    def moni_check(self):
        if time.time() > self.next_check_stat_time:
            self.send_moni()
            self.next_check_stat_time += 5
            self.receive_data_count = 0

    def send_moni(self):
        self.moni.record({'inf_server_receive_instance': self.receive_data_count})
        result = self.moni.result(msg_type='inf_server')
        if result != {}:
            result['ip'] = self.ip
            self.zmq_adaptor.logger_sender.send(pickle.dumps(result))
            self.logger.info('send moni to log')

    def run(self):
        session = tf.Session(config=self.config)
        self.tf_session = session
        self.model.tf_session = session
        self.model.tf_session.run(tf.global_variables_initializer())

        socks = dict(self.zmq_adaptor.poller.poll())
        self.receive_model(socks)
        self.logger.info("inference_server latest model ready.")
        while True:
            socks = dict(self.zmq_adaptor.poller.poll(timeout=0.01))
            self.receive_model(socks)
            self.receive_data(socks)
            if self.training_set.len() > 0 and time.time() > self.next_check_time:
                self.next_check_time = time.time() + self.max_waiting_time
                s_time = time.time()
                self.logger.info("start convert_to_np, count {0}, gpu {1}".format(self.training_set.len(), self.gpu_num))
                data = self.training_set.convert2np()
                self.logger.info("start get_predict_action_batch, cost {0}".format(time.time() - s_time))
                raw_result = self.model.predict_batch(data)
                # TODO bencmark this code
                result_list = raw_result.reshape((-1, self.player_num, raw_result.shape[-1]))
                for index, m in enumerate(self.raw_data_list):
                    m[-1] = pickle.dumps(result_list[index])
                    self.zmq_adaptor.router_receiver.send_multipart(m)
                self.raw_data_list = []
                self.training_set.clear()
                self.logger.info("send back finish, cost: {0}".format(time.time() - s_time))
            self.moni_check()
