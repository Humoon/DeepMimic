from train.common.config import is_development
import os, random, shutil, sys, copy
from loguru import logger
import numpy as np


def openai_sample(pd):
    noise = np.random.uniform(pd + 1e-8)
    mask = np.where(pd == 0.0, 0, 1)
    pd_with_noise = pd - np.log(-np.log(noise))
    return np.argmax(pd_with_noise * mask)


def distribution_sampling(distribution, index_range_list=[]):
    r = random.randint(0, 10000) + 1e-5
    accumulated_prob = 0
    last_positive_value_index = -1
    if len(index_range_list) == 0:
        index_range = range(len(distribution))
    else:
        index_range = index_range_list
    for index in index_range:
        i = distribution[index]
        if i > 1e-10:
            last_positive_value_index = index
        accumulated_prob += 10000 * i
        if accumulated_prob >= r:
            return index
    return last_positive_value_index


def multi_distribution_sampling(distribution):
    actions = []
    for pd in distribution:
        r = random.randint(0, 10000) + 1e-5
        accumulated_prob = 0
        last_positive_value_index = -1

        for index in range(len(pd)):
            prob = pd[index]
            if prob > 1e-10:
                last_positive_value_index = index
            accumulated_prob += 10000 * prob
            if accumulated_prob >= r:
                actions.append(index)
                break

            if index == len(pd) - 1:
                actions.append(last_positive_value_index)
    return actions


def get_max_diff(list_a):
    s_a = sorted(list_a)
    if len(s_a) > 2 and s_a[-2] > 0:
        diff_percent = (s_a[-1] - s_a[-2]) / s_a[-2]
    else:
        # get max index
        diff_percent = 1

    return diff_percent > 0.5 and s_a[-1] > 0.4


def create_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def setup_logger(log_file, level=None):
    if level is None:
        if is_development():
            level = "DEBUG"
            logger.add(
                f"log/{log_file}",
                format="{time} {level} {message}",
                level=level,
                rotation="4 GB",
                retention="0.5 days",
                mode="w+")
            return logger
        else:
            level = "INFO"

    logger.remove()
    context_logger = copy.deepcopy(logger)
    context_logger.add(
        f"log/{log_file}", format="{time} {level} {message}", level=level, rotation="4 GB", retention="0.5 days", mode="w+")
    context_logger.add(sys.stderr, colorize=True, format="{time} {level} {message}", level=level)
    context_logger.add(f"log/{log_file}.error", format="{time} {level} {message}", backtrace=True, diagnose=True, level="ERROR")
    return context_logger
