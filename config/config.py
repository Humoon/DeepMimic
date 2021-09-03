import yaml
import os


# 评估配置
def evaluation_config():
    return read_config("config/evaluation.yaml")


# redis config
def redis_config():
    return read_config("config/redis.yaml")


# 读取配置文件
def read_config(filepath):
    res = None
    with open(filepath, "rt", encoding="utf8") as f:
        res = yaml.safe_load(f)
    return res[enviroment()]


def enviroment():
    return "production" if is_production() else "development"


# 判断是否是生产环境
def is_production():
    return not is_development()


# 判断是否是开发环境
def is_development():
    return "LOCALTEST" in os.environ.keys()
