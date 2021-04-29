from matplotlib.pyplot import xcorr
import torch as th
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from .epsilon_schedules import DecayThenFlatSchedule

class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, probs=None, temperature=1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -th.log( -th.log( U + self.eps))

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()

def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy()

REGISTRY = {}

class GumbelSoftmaxMultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_logits, avail_actions, t_env, test_mode=False):
        masked_policies = agent_logits.clone()
        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = GumbelSoftmax(logits=masked_policies).sample()
            picked_actions = th.argmax(picked_actions, dim=-1).long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions


REGISTRY["gumbel"] = GumbelSoftmaxMultinomialActionSelector


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0] = 0
        masked_policies = masked_policies / (masked_policies.sum(-1, keepdim=True) + 1e-8)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            self.epsilon = self.schedule.eval(t_env)

            epsilon_action_num = (avail_actions.sum(-1, keepdim=True) + 1e-8)
            masked_policies = ((1 - self.epsilon) * masked_policies
                        + avail_actions * self.epsilon/epsilon_action_num)
            masked_policies[avail_actions == 0] = 0
            
            picked_actions = Categorical(masked_policies).sample().long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector

def categorical_entropy(probs):
    assert probs.size(-1) > 1
    return Categorical(probs=probs).entropy()

class DiscreteDeterministicActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0] = 0
        masked_policies = masked_policies / (masked_policies.sum(-1, keepdim=True) + 1e-8)

        # greedy
        picked_actions = masked_policies.max(dim=2)[1]
        masked_policies = th.zeros_like(masked_policies).scatter_(2, picked_actions.unsqueeze(-1), 1)

        if not test_mode or not self.test_greedy:
            self.epsilon = self.schedule.eval(t_env)
            
            epsilon_action_num = avail_actions.sum(-1, keepdim=True)
            masked_policies = ((1 - self.epsilon) * masked_policies
                        + avail_actions * self.epsilon/epsilon_action_num)
            masked_policies[avail_actions == 0] = 0
            
            picked_actions = Categorical(masked_policies).sample().long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions


REGISTRY["discrete_deterministic"] = DiscreteDeterministicActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """
        根据Q值选择动作
        :param agent_inputs:
        :type agent_inputs: shape: torch.Size([8, 8, 6])
        :param avail_actions:
        :type avail_actions:  shape: torch.Size([8, 8, 6])
        :param t_env:  eg: 0
        :type t_env: int
        :param test_mode:
        :type test_mode:
        :return:
        :rtype:
        """
        # 假设Agent_Inputs是每个agent bav的一个batch的Q值 ，计算epsilon随着时间步的decay后的值
        self.epsilon = self.schedule.eval(t_env)
        # 如果测试模式, 不要随机探索，epsilon设为0
        if test_mode:
            # 只有贪婪的行动选择
            self.epsilon = 0.0

        # 从选择中排除的mask操作
        masked_q_values = agent_inputs.clone()
        # 永远不应该被选中！
        masked_q_values[avail_actions == 0] = -float("inf")
        
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class GaussianActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, mu, sigma, test_mode=False):
        # Expects the following input dimensions:
        # mu: [b x a x u]
        # sigma: [b x a x u x u]
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"
        sigma = sigma.view(-1, self.args.n_agents, self.args.n_actions, self.args.n_actions)

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            dst = th.distributions.MultivariateNormal(mu.view(-1,
                                                              mu.shape[-1]),
                                                      sigma.view(-1,
                                                                 mu.shape[-1],
                                                                 mu.shape[-1]))
            try:
                picked_actions = dst.sample().view(*mu.shape)
            except Exception as e:
                a = 5
                pass
        return picked_actions


REGISTRY["gaussian"] = GaussianActionSelector