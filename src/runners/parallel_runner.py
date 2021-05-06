from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    # 并发运行
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # eg: 8
        self.batch_size = self.args.batch_size_run

        # 为envs制作子进程, 并发进行
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for worker_conn in self.worker_conns:
            ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)
        # ps里面有8个进程, 启动每个进程
        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()   #eg: {'state_shape': 300, 'obs_shape': 75, 'n_actions': 6, 'n_agents': 8, 'episode_limit': 200}
        self.episode_limit = self.env_info["episode_limit"]  #eg: 200
        # 时间步，记录
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # 对于多进程的每个环境，都进行reset
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        # 初始化一个pre_transition_data，存放每个agent返回的信息，state，obs， actions
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # 获取返回的obs，State和Avail_actions
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        # 更新数据到buffer中
        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        # 重置环境
        self.reset()
        all_terminated = False
        # 初始化episod的奖励，默认[0, 0, 0, 0, 0, 0, 0, 0]
        episode_returns = [0 for _ in range(self.batch_size)]
        # episode进行了多少个小的回合，初始化[0, 0, 0, 0, 0, 0, 0, 0]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # 可以存储额外的统计资料，如战斗胜利，这是按终止顺序填写的。
        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # 将到目前为止的整批经验传递给agents
            # 在这个时间段内，以批次的方式接收每个agent的行动，对于每个未结束的env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            # action转到cpu
            cpu_actions = actions.to("cpu").numpy()

            # 更新所采取的操作，
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            #把action更新到buffer中
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 向每个env发送action
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # 我们为此env制作了行动
                    if not terminated[idx]: # 如果它没有结束，将动作发送到ENV
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # 更新 envs_not_terminated，判断
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            # 判断是否所有env都结束了
            all_terminated = all(terminated)
            #如果都结束，退出
            if all_terminated:
                break

            # 发送step数据，我们将插入当前的timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # 下一步我们将插入数据，以便选择一个action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            #为每个未结束的env接收数据
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    # data ：dict 包括'state', 'avail_actions','obs','reward','terminated','info'
                    data = parent_conn.recv()
                    # 当前时间的剩余数据
                    post_transition_data["reward"].append((data["reward"],))
                    # 这个episode的每个agent返回的奖励
                    episode_returns[idx] += data["reward"]
                    # 每个agnent的进行了多少个episode，记录下
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # 下一个时间步选择一个action的数据
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 时间步+1
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # 为每个env获取统计数据
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    """
    创建环境，env_fn是初始化环境的配置信息
    :param remote: eg: <multiprocessing.connection.Connection object at 0x7fc6f40d3dc0>
    :type remote:
    :param env_fn: eg: <runners.parallel_runner.CloudpickleWrapper object at 0x7fc6f828d8e0>
    :type env_fn:
    :return:
    :rtype:
    """
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    使用CloudPickle序列化内容（否则多处理尝试使用pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

