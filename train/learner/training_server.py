import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import zmq, time, pickle, sys
from train.learner.training_set import TrainingSet
from train.common.utils import create_path, setup_logger
from net.zmq_adaptor import ZmqAdaptor
from train.summary.moni import Moni
from train.common.config import is_development, read_config
from train.discovery.name_server_api import NameServerAPI
import importlib
import pyarrow.plasma as plasma
import copy


class TrainingServer:

    def __init__(self, use_distribution=False):
        self.config = tf.ConfigProto()
        self.mode = "football"
        self.test = is_development()
        self.mpi_rank = 0
        self.rank = 0
        self.gpu_num = 0

        param = read_config("config/ppo_train.yaml")
        self.player_num = param["player_num"]
        self.batch_size = param["batch_size"]

        pull_ns_api = NameServerAPI()

        if use_distribution is True:
            import horovod.tensorflow as hvd

            hvd.init()
            res, self.gpu_num = pull_ns_api.register_gpu()
            self.config.gpu_options.visible_device_list = str(self.gpu_num)
            self.config.gpu_options.allow_growth = True
            self.mpi_rank = hvd.local_rank()
            self.rank = hvd.rank()
            self.hooks = [hvd.BroadcastGlobalVariablesHook(0)]

        self.checkpoint_dir = "./checkpoints" if self.rank == 0 else None

        if is_development():
            self.plasma_client = plasma.connect("/tmp/plasma", 2)
        else:
            self.plasma_client = plasma.connect("/tmp/plasma/plasma", 2)
        self.data_id = plasma.ObjectID(20 * bytes(str(self.mpi_rank), encoding="utf-8"))

        need_dirs = [
            "./log/saved_model_%s_%d/" % (self.mode, self.mpi_rank),
            "./log/gpu_server_log_%s_%d/" % (self.mode, self.mpi_rank)
        ]
        for d in need_dirs:
            create_path(d)
        self.logger = setup_logger(log_file="./gpu_server_log_%s_%d/log" % (self.mode, self.mpi_rank))

        # net server
        self.zmq_adaptor = ZmqAdaptor(logger=self.logger)

        # publish model
        if self.rank == 0:
            pub_ns_api = NameServerAPI()
            _, port = pub_ns_api.register(rtype="pub_model_server", extra={"data_type": "train", "zmq_mode": "pub"})
            self.zmq_adaptor.start({"mode": "pub", "host": "*", "port": port})

        # net client to log server
        _res, log_services = pull_ns_api.discovery_log_server(block=True)
        self.zmq_adaptor.start({
            "mode": "push",
            "host": log_services[0].address,
            "port": log_services[0].port,
            "dest": "logger"
        })

        ppo_model = importlib.import_module("model.football_models." + param['football_model'])
        self.model = ppo_model.PPOModel(param, use_distribution=use_distribution, training_server=True)
        self.sess = None # tf.Session()
        self.last_sgd_time = time.time()
        self.start_training = False

        #  Moni
        self.next_save_model_time = time.time() + 60 * 20
        self.model_time = 0 # 保存模型的时间戳
        self.next_check_stat_time = time.time() + 60
        self.receive_instance_num = 0
        self.wait_data_server_count = 0

        self.model_moni = Moni()
        self.train_moni = Moni()

    def moni_check(self):
        if time.time() > self.next_check_stat_time:
            # 每10秒发一次数据
            self.send_moni(msg_type="training_server")
            self.next_check_stat_time += 10
            self.logger.info("send moni to log")

    def send_moni(self, msg_type=None):
        if msg_type == "training_server":
            result = self.train_moni.result(msg_type=msg_type)
        elif msg_type == "model":
            result = self.model_moni.result(msg_type=msg_type)
        else:
            result = {}
        if result != {}:
            self.zmq_adaptor.logger_sender.send(pickle.dumps(result))

    def run(self):
        # signal.signal(signal.SIGCHLD, signal.SIG_IGN)

        if self.test:
            session = tf.Session()
            with session.as_default():
                tf.global_variables_initializer().run()
        else:
            session = tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.checkpoint_dir, hooks=self.hooks, config=self.config)

        self.tf_session = session
        self.model.tf_session = session
        # self.model.restore_ckp_model('./checkpoints/')
        # self.log_basic_info.info('load model')
        # mt=self.model.serializing_with_session()

        if self.rank == 0:
            self.pub_model()

        training_end_time = time.time()
        while True:
            if time.time() > self.next_save_model_time and not self.test and self.rank == 0:
                self.save_model()
                self.next_save_model_time += 60 * 20
            # root gpu server decide start of training
            batch_data = self.sample_data()
            if batch_data is None:
                time.sleep(0.01)
                self.logger.info(f"waiting for data server ready...")
                self.wait_data_server_count += 1
                continue
            else:
                self.train_moni.record({"receive_data_time": time.time() - training_end_time})
            self.learn(batch_data)
            self.train_moni.record({
                "total_training_time": time.time() - training_end_time,
                "wait_data_server_count": self.wait_data_server_count
            })
            training_end_time = time.time()
            self.plasma_client.delete([self.data_id])
            self.moni_check()
            self.wait_data_server_count = 0

    def sample_data(self):
        if self.plasma_client.contains(self.data_id):
            data = self.plasma_client.get(self.data_id, timeout_ms=0)
            batch_data = copy.deepcopy(data)
            # self.plasma_client.delete([self.data_id])
            self.logger.info(f"training server got batch data and delete object index {self.mpi_rank}")
            return batch_data
        else:
            return None

    def learn(self, batch_data):
        receive_instance_num = batch_data.pop("receive_instance_num")
        start_time = time.time()
        model_log_dict, train_log_dict = self.model.learn(
            batch_data,
            train_server_logger=self.logger,
            mpi_rank=self.rank,
        )
        end_time = time.time()
        self.logger.info("training time: {0}".format(end_time - start_time))
        if self.rank == 0:
            self.pub_model()

        model_log_dict.update({
            "Q_value": np.mean(batch_data["q_reward"]),
            "advantage_value": np.mean(batch_data["gae_advantage"]),
        })

        self.model_moni.record(model_log_dict)
        self.send_moni(msg_type="model")

        train_log_dict.update({
            "training_steps_total": self.model.update_times,
            "policy_different": self.model.update_times - np.mean(batch_data["model_update_times_list"]),
            "action_entropy_parameter": self.model.action_entropy_parameter,
            "learning_rate": self.model.learning_rate,
            "receive_instance_total": receive_instance_num,
            f"receive_instance_rank_{self.rank}": receive_instance_num,
            "data_efficiency": receive_instance_num / self.batch_size,
            "training_time": end_time - start_time,
        })
        self.train_moni.record(train_log_dict)

    def pub_model(self):
        model_str, self.model_time = self.model.serializing(update_times=self.model.update_times, logger=self.logger)
        self.logger.debug("root gpu server pub model, model size {0}, mode {1}".format(sys.getsizeof(model_str), self.mode))
        # 只通过 zmq 传给 worker gpu
        self.zmq_adaptor.publisher.send(model_str)

    def save_model(self):
        bt = time.time()
        model_str, self.model_time = self.model.serializing(update_times=self.model.update_times, logger=self.logger)
        model_file = "./log/saved_model_{0}_{1}/{0}_model_f".format(self.mode, self.mpi_rank) + "_" + str(self.model_time)
        with open(model_file, "wb") as f:
            f.write(model_str)
        self.logger.info("save model, use {0}, mode {1}".format(time.time() - bt, self.mode))
