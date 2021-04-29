from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# 此多multi-agent控制器共享agent之间的参数  This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """

        :param ep_batch:   eg: EpisodeBatch. Batch Size:8 Max_seq_len:201 Keys:dict_keys(['state', 'obs', 'actions', 'avail_actions', 'probs', 'reward', 'terminated', 'actions_onehot', 'filled']) Groups:dict_keys(['agents'])
        :type ep_batch:
        :param t_ep:   eg: 0
        :type t_ep:
        :param t_env:  eg: 0
        :type t_env:
        :param bs:
        :type bs:   eg: [0, 1, 2, 3, 4, 5, 6, 7]
        :param test_mode:
        :type test_mode:  bool
        :return:
        :rtype:
        """
        # 仅在bs中选择所选批次元素的操作, eg:
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # Q值 qvals: shape : torch.Size([8, 8, 6])
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        #
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """
        前向
        :param ep_batch: epsiode batch 信息
        :type ep_batch:
        :param t:   时间步, timestamp
        :type t:  int
        :param test_mode:
        :type test_mode:
        :return:
        :rtype:
        """
        # 从buffer中提取时间步t对应的输入
        agent_inputs = self._build_inputs(ep_batch, t)
        # 提取对应的actions
        avail_actions = ep_batch["avail_actions"][:, t]
        # 根据隐藏状态和输入，放入rnn模型，获取rnn的输出和新的隐藏层状态
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs