import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import REGISTRY as run_REGISTRY
# 如果您想在控制台中看到stdout / stderr，则设置为“no”，保存到文件设置为 "fd"
SETTINGS['CAPTURE_MODE'] = "no"
logger = get_logger()
# 初始化一个管理配置的应用ex
ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    """
   主函数
    :param _run:  scare 管理的_run
    :type _run:
    :param _config:scare 管理的_config
    :type _config:
    :param _log:
    :type _log:scare 管理的_log
    :return:
    :rtype:
    """
    # 拷贝出一份配置
    config = config_copy(_config)
    # 在整个module中设置随机种子
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    
    #调用run/run.py 中的run函数
    run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    """
    从给定命令行参数params找到config，加载环境或算法的配置
    :param params:
    :type params:
    :param arg_name:  命令行参数params的对应的 'env_args.map_name = stag_hunt'
    :type arg_name:
    :param subfolder: 配置文件的目录
    :type subfolder:
    :return:
    :rtype:
    """
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    """
    把字典u中的内容合并到字典d中
    :param d:  配置信息，字典d
    :type d:
    :param u:  配置信息，字典u
    :type u:
    :return: 合并后的字典
    :rtype:
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    """
    copy出一份给定的config
    :param config:
    :type config:
    :return:返回拷贝的config
    :rtype:
    """
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    """
    获取命令参数传过来的值
    :param params:  eg: ['/Users/admin/git/pymarl2/src/main.py', 'with', 'env_args.map_name=stag_hunt']
    :type params:
    :param key:  eg: 'env_args.map_name'
    :type key:
    :param default:  eg: 'stag_hunt'
    :type default:
    :return:  'stag_hunt'
    :rtype:
    """
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    #  命令行的参数获取
    params = deepcopy(sys.argv)
    # 默认值的获取default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # 加载算法和env基础配置, 如果params是stag_hunt，就会加载配置 src/config/envs/stag_hunt.yaml
    env_config = _get_config(params, "--env-config", "envs")
    # 如果params中的算法参数是'--config=qmix_prey' ，就会加载src/config/algs/qmix_prey.yaml
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    # 合并env配置到我们的主要配置中 config_dict
    config_dict = recursive_dict_update(config_dict, env_config)
    #  合并算法配置到我们主要配置中config_dict
    config_dict = recursive_dict_update(config_dict, alg_config)

    # 现在将所有配置添加到sacred, 配置管理器ex
    ex.add_config(config_dict)

    # 将默认为Sacred保存到磁盘
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name']) 
    file_obs_path = join(results_path, "sacred", map_name, algo_name)
    
    logger.info("保存配置文件 FileStorageObserver 到 {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # 开始运行main函数 my_main
    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
