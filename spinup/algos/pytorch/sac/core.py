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

# sizes是一个列表，决定了神经网络每一层神经元的数量，len(sizes)决定了网络的深度。e.g. sizes = [10, 20, 5]，则神经网络有三层，分别有 10、20 和 5 个神经元。
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1): #使用该循环遍历sizes中的元素，以此创建网络
        act = activation if j < len(sizes)-2 else output_activation # 检查是否是最后一层，因为只有最后一层的激活函数不一样
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()] # 每一层有一个全连接+一个激活函数，全连接的输入为sizes[j]，输出为sizes[j+1]，换句话说，本层有sizes[j]个神经元，下层有sizes[j+1]个。下一层的输入数量必须等于上一层的输出，而这个for循环可以做到。
    return nn.Sequential(*layers) #返回一个nn sequential。
    # 在 Python 中，* 操作符用于解包（unpacking）。在这个上下文中，*layers 将列表 layers 中的元素解包，作为 nn.Sequential() 函数的参数传递进去。nn.Sequential() 函数期望的是一系列的模块作为参数，而不是一个包含模块的列表。通过使用 * 操作符，可以将列表中的元素一个一个地传递给 nn.Sequential()，而不是将整个列表作为一个单独的参数传递。例如，如果 layers 是一个包含三个模块的列表：[module1, module2, module3]，那么 nn.Sequential(*layers) 等价于 nn.Sequential(module1, module2, module3)。这种解包操作是为了方便地将一个列表或元组中的元素传递给函数或构造函数，而不必手动一个一个地列出。在这里，它确保 nn.Sequential() 接收的是多个模块而不是一个包含模块的列表。

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()]) 
    # np.prod是NumPy库中的函数，用于计算数组元素的乘积。在这里，它被用于计算参数p的形状中所有维度的乘积，即参数的总元素数量。
    # 这样来看，首先 for p in module.parameters() 拿出了网络中的每一层的可学习参数，p.shape可能是(input_dim,output_dim)，这样使用prod乘起来就是
    # 整层网络的全部参数，然后用sum加起来就是整个网络的全部可学习参数了。


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation) # 这里与Q网络初始化方式一样
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim) 
        self.act_limit = act_limit
        # 回想SAC，他输出的是动作的均值和方差，那么self.mu_layer与self.log_std_layer就很好理解了，首先输出均值和方差是两个网络，他们共享特征提取层，只不过最后一层不一样，当然他们的定义其实是一样的，都是一层输入为hidden_sizes[-1]（即隐藏层最后一层的大小），输出为action维度的一个vector。这一点原来不理解，一看代码就很好理解了，对于每一个维度都输出一个属于这个维度的均值和方差。

    # recall：forward函数定义数据在网络中的正向传播方式
    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs) # state首先传入共享的特征提取层，输出net_out
        mu = self.mu_layer(net_out) # mu的size等于act_dim.
        log_std = self.log_std_layer(net_out) # net_out分别输入均值方差对应的最后一层网络，从而得到动作函数的均值和方差
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # 相当于一个截断函数，将方差限制在固定范围内。
        std = torch.exp(log_std)
        # ？？？问题：为什么mu不需要截断？ANS：在这个实现中，通过 torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) 对方差进行截断，是为了确保方差的数值稳定性。这是因为方差必须是正数，而且太大或太小的方差都可能导致数值计算上的问题。对于均值而言，通常不需要进行类似的截断，因为均值的取值范围一般可以是实数的任意值。对于连续动作空间，使用 Tanh 函数将均值映射到 [-1, 1] 的范围内，然后再通过 self.act_limit * pi_action 将均值映射到动作空间的具体范围。
        
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            # 看到这个地方就明白了为什么会在SAC中存在deterministic policy的情况，训练的过程中，为了保留策略的探索性，需要使用策略的均值和方差；当在测试阶段，需要的还是一个确定的策略，因此只拿到均值就够了。当然这个地方也可以不做区分，但是为了避免数据无意义的计算和流动，如果不需要使用方差，就干脆不要去计算它。
            # 换句话说，其实在SAC中就讨论过那个问题，就是探索性是在训练过程中需要的，而不是最优策略需要的。
            # 没有必要（甚至不可能）存在一个呈现分布式的最优策略，也就是说，你不要担心使用DRL的时候，就需要策略在最后阶段也是呈现分布式的。
            # 只需要策略函数在训练过程中能保留多峰性，能对几个比较重要的地方都以概率进行探索就行，目前来看，之前的一些顾虑是不存在的。DRL理论框架还是自洽的。
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        # 注意这个地方pi_action从Normal(mu, std)中进行了重采样，不再等于mu了。

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1) ###这个地方可能是把pi_action对应的所有维度的log_probability求和了。
            # 但是不论怎样，想要计算logpi(pi_action)这个值，都必须知道pi的真实分布，所以必须假设为gaussian，
            # 否则重采样之后，如果策略分布的pdf未知将无法计算这个值。
            # ！！！你可能会想，pi(a|s)中的given “s”是如何体现的。ANS：上面的Normal(mu, std)中的mu与std，就是从上面obs输入神经网络得来的
            # 所以说，pi(.|s)的策略分布就是Normal(mu, std)，这个分布中是蕴含着s信息的。
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            # 这个地方的解释比较tricky，你可以理解为将计算到的logp_pi减去一个修正项，从而更有利于数值的稳定。具体的数学原理，建议去参考原文appendix。
            # 至于数值稳定性，指的是让系统对数值变化、计算稳定，不要出现什么数值问题。
        else:
            logp_pi = None
        # 注意这里计算得到logp_pi之后就等着返回了，下面两行不再使用logp_pi

        pi_action = torch.tanh(pi_action) #  使用 Tanh 函数将原始的策略网络输出的动作 pi_action 映射到 [-1, 1] 的范围。
        pi_action = self.act_limit * pi_action # 上面讲action映射到[-1，1]，这里再乘上self.act_limit，从将pi_action映射到动作空间。

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        # 现在理解了上面的mlp函数，这个self.q也就很清楚了，mlp的输入为[obs_dim + act_dim] + list(hidden_sizes) + [1]，首先，Q(s,a)中s与a的输入就是直接连在一起组成一个vector，当然你也可以理解成第一层有（obs_dim + act_dim）个神经元，list(hidden_sizes)是一个list，e.g. [20,50,20]，具体效果就如上面mlp所示，最后一层只有一个神经元，输出的是Q(s,a)的值。

    def forward(self, obs, act): # 还记得，forward（前向过程）是将输入放进网络，得到输出的过程。
        q = self.q(torch.cat([obs, act], dim=-1)) # dim=-1是cat的连接方式，正式的说法是按照哪个维度连接，直观的理解是对于两个vector横着连接还是竖着连接
        return torch.squeeze(q, -1) # Critical to ensure q has right shape. torch.squeeze(q, -1) 这一行代码实际上是为了确保输出是一个标量而不是一个形状为 (1,) 的张量。

# 这个class属于为actor critic选择网络结构，还是属于比较高层的网络，其调用的函数才是网络的具体结构。
class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions。注意这里的self.pi，q1，q2都是类，调用其中的forward函数才进行前向传播。
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

# ##总结：这个core文件主要是定义了actor-critic网络结构，actor-critic继承自MLPActorCritic类，而这个类使用了上面给出的MLP和MLPQ两个function定义网络结构。
# 因此在sac算法代码中使用到actor-critic命令之后，所有的网络就都已经创建好了，并且包括了其中数据流动的方向。
