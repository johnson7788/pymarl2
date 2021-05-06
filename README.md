# RIIT
我们为[RIIT：重新思考多AgentReinforcement Learning中实施技巧的重要性](https://arxiv.org/abs/2102.03479)提供了开源的代码。
我们实现了众多实现SOTA的QMIX变体算法的超参数，并对其进行了标准化。

## Python MARL framework
PyMARL是[WhiRL](http://whirl.cs.ox.ac.uk)的深度多agent强化学习框架，包括以下算法的实现。

Value-based Methods:

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**MAVEN**: MAVEN: Multi-Agent Variational Exploration](https://arxiv.org/abs/1910.07483)
- [**Qatten**: Qatten: A general framework for cooperative multiagent reinforcement learning](https://arxiv.org/abs/2002.03939)
- [**QPLEX**: Qplex: Duplex dueling multi-agent q-learning](https://arxiv.org/abs/2008.01062)
- [**WQMIX**: Weighted QMIX: Expanding Monotonic Value Function Factorisation](https://arxiv.org/abs/2006.10800)

Actor Critic Methods:

- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VMIX**: Value-Decomposition Multi-Agent Actor-Critics](https://arxiv.org/abs/2007.12306)
- [**FacMADDPG**: Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://arxiv.org/abs/2003.06709)
- [**LICA**: Learning Implicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2007.02529)
- [**DOP**: Off-Policy Multi-Agent Decomposed Policy Gradients](https://arxiv.org/abs/2007.12322)
- [**RIIT**: RIIT: Rethinking the Importance of Implementation Tricks in Multi-AgentReinforcement Learning](https://arxiv.org/abs/2102.03479)

PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## 安装说明 

Install Python packages
```shell
# require Anaconda 3 or Miniconda 3
bash install_dependecies.sh
```

如果不用sc2环境可以不安装, 设置星际争霸II和SMAC： StarCraft Multi-Agent Challenge
```shell
bash install_sc2.sh
```

这将把StarCraft II下载到第三方文件夹，并复制必要的地图来运行。

## 运行实验 

```shell
# SMAC StarCraft Multi-Agent Challenge,  对于多agent的StarCraft
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```

```shell
# 适用于合作捕食者-猎物： Cooperative Predator-Prey 
python3 src/main.py --config=qmix_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```

配置文件作为一种算法或环境的默认值。

它们都位于`src/config`中。
`--config`指的是`src/config/algs`中的配置文件。
`--env-config`指的是`src/config/envs`中的配置文件。

## 代码的流程
parallel_runner.py 中的env_worker 负责初始化环境,---加载src/envs/__init__.py中的env_fn加载环境

## 运行并行实验： 
```shell
# bash run.sh config_name map_name_list (threads_num arg_list gpu_list experinments_num)
bash run.sh qmix corridor 2 epsilon_anneal_time=500000 0,1 5
```

`xxx_list` is separated by `,`.

所有的结果将被存储在`Results`文件夹中，并以`map_name`命名。

## 强制所有训练进程退出 

```shell
# 当前用户的所有Python和游戏进程将退出。
bash clean.sh
```

## 一些测试结果超级困难的游戏情景 
![](img/baselines2.png)

## Cite
```
@article{hu2021riit,
      title={RIIT: Rethinking the Importance of Implementation Tricks in Multi-Agent Reinforcement Learning}, 
      author={Jian Hu and Siyang Jiang and Seth Austin Harding and Haibin Wu and Shih-wei Liao},
      year={2021},
      eprint={2102.03479},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

