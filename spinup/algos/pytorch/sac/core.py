import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# 这个core文件代码，主要定义了（1）不影响算法理解，但需要使用的操作（2）神经网络的结构，因此在alg文件中，只要一行命令就可以创建一整个价值网络和策略网络，而网络的torch结构，在这个文件中定义。之所以要读代码，也是让你学习一下强化学习代码写作的文件格式，不要一个文件放下所有的东西。

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape) 
    #isscalar检查shape是实数还是别的，是实数返回(length,shape)。When * is used in a function call or a tuple creation, it allows you to unpack（打包） the elements of an iterable (such as a list or tuple) and include them as separate elements in the resulting sequence. 举例来说length = 3 shape = (4, 5) result = (length, *shape)； the value of result would be a tuple (3, 4, 5)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act): # 还记得，forward（前向过程）是将输入放进网络，得到输出的过程。
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

# 这个class属于为actor critic选择网络结构，还是属于比较高层的网络，其调用的函数才是网络的具体结构。
class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

# 这里定义的是得到动作的函数，就是把obs放到了pi网络里面。deterministic看SquashedGaussianMLPActor，这个是具体的网络，spinningup将每个不同的网络结构封装成类。
    def act(self, obs, deterministic=False):
        with torch.no_grad(): # 这个是常见且关键的操作，详细解释见下面。
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
# 在PyTorch中，默认情况下，每个操作都会跟踪其输入的梯度，以便在反向传播时计算梯度。这种机制是为了支持自动微分，从而可以方便地进行梯度下降优化。但是这回占用额外的内存和计算资源，因此如果在执行神经网络的某个地方确切地知道你不需要计算梯度，使用 with torch.no_grad(): 是一种良好的实践。这可以降低内存消耗和计算开销，特别是在推理阶段或者其他不需要梯度信息的计算中。
# 而这个地方，我们只是为了从策略网络拿到具体的动作，是不需要梯度的。！！！这个东西应该是很经验的，只有你看了多了代码才能确保自己不会写错，不要乱加。
