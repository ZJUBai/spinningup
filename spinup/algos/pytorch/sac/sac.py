from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
# 定义replay buffer，初始化函数输入state，action大小和size变量，用来决定buffer中矩阵的行与列的数量，buffer class中每一个小的子buffer，存放size条state与action。
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) #因为obs和action本身数据结构可能复杂，所以是一个矩阵
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) #指定数据类型可以避免许多数值问题，免得默认的数据类型不一样导致bug
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32) #reward和done都是一个值所以不需要存为矩阵，只要一维即可。
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        
# 存储函数输入一个转移的全部数据，然后会存到buffer中，buffer是当前存储的全部数据，buffer之外的数据会被丢弃
    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs #self.ptr是一个计数器，对应着现在是存第几个数据
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size #如果ptr+1的大小超过了maxsize，则会从前向后覆盖replay buffer
        self.size = min(self.size+1, self.max_size) #selfsize是在记录当前buffer的大小，如果buffer满了其大小不再增长，selfsize将一直等于maxsize

# batch的大小远小于buffer，且batch从buffer生成，所以要先定义buffer，再定义batch。
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size) # It generates random integers between 0 (inclusive) and self.size (exclusive). The size parameter determines the number of integers to generate, and in this case, it is set to batch_size. 也就是说，idxs生成了一个buffer需要存放的数据对应的坐标，如果buffer满了，将从0到buffersize之间生成32个数据组成idxs，其为一个nparray。
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs]) #将batch存成一个字典，每一个batch中包含'obs': torch.Tensor(...)等5个键值对
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()} # 这个return的目的是进行一个数据格式的转换，其实也可以直接return batch，但是一般使用到batch的时候就准备进行SGD了，因此直接将np数据转换为torch tensor。要理解其返回的依然是一个字典，只不过value的值变成了torch tensor。

# 这个是SAC算法的主函数，像spinningup这种较为简单的框架，会把一个算法所有的东西都写成一个函数，这样比较易于理解。
# 这其中比较重要的有env_fn：决定了跑哪个环境，这个东西必须满足OpenAI Gym的API，比如说可以丢进去一个action，然后用step返回下一帧的信息等。如果要换成不满足gym API的环境，（可能）最简单的方式是，写一个壳子，让那个环境可以向Gym一样被调用，而不是将sac中的数据结构改成新环境的数据结构。
# actor_critic:创建sac算法中需要用到的网络。actor_critic=core.MLPActorCritic这个是一乐MLP类，相当于创建了actor网络和critic网络。所以要修改网络结构，需要看core.MLPActorCritic这个类，同时ac_kwargs=dict()是输入网络相关参数的，同样需要看这个类。这个框架默认使用的是最传统的三层网络结构，你可把网络参数写成输入从而方便的调节网络，但是建议你要么先实验，要么参考别人的网络结构，不然搜索的空间将过大。
# ac_kwargs=dict()与logger_kwargs=dict()这种把超参做成字典作为输入，一般对应一个比较大的算法子模块，如果把字典里面的超参拆开输入也可以，但是一般都不是实验关心的核心超参，所以干脆封装成一个字典输入，这样会比较干净，甚至可以把“seed=0, steps_per_epoch=4000, epochs=100...”全部封装成一个字典作为输入。
# polyak,目标网络修改的频率，举例来说，目标网络会按照一定步长做修改，而不是直接复制价值网络，polyak就决定修改的幅度。
def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # 这里先定义存储的log文件，Epochlogger是从utils中import的一个class。
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed) # 这里设置随机种子，torch与np中的随机性会由各自的随机种子决定。不是所有函数都有返回，torch是一个非常大的类，中间有很多self.para的东西，相当于你传入了一个参数，就存放torch中，直到下次有代码覆盖这个值，这个值就一直不会变，而当需要用到的时候，就会自动调用这里面存放的值。
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn() # env_fn()是输入的环境，这个地方与gym的API不完全相同，不过基本可以理解成env = gym.make("CartPole-v1")，这里env_fn就相当于gym.make，因为算法与环境是分开的，所以不会写成gym.make。
    obs_dim = env.observation_space.shape # observation其实本质上不需要区分连续与离散，因为不论游戏展现出来的是围棋还是星际争霸，其实对于RL都是离散环境，都是输入一个action，返回下一state，因此state需要注意的是是否是图象输入，而action需要注意是否连续动作。
    act_dim = env.action_space.shape[0] # 这个地方就要开始注意环境是连续动作空间还是离散动作空间，如果是连续动作空间，shape【0】会拿到动作的维度，如果是离散空间，动作的的返回是离散动作的数量，这个地方还是要具体看Gym的API。
    # state是否图象，action是否连续是会关系到相应网络结构的。

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # 首先这里假设是连续的，那么env.action_space.high[0]将拿到action第一个维度的upper bound，但是作为所有action维度的bound确实可能带来bug，需要确认是不是真的所有action的范围都是（-1，1）！经过核实，Mujoco的所有环境都把动作归一化到了固定范围，虽然不一定是（-1，1），但所有action的维度都是一样的。
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks，开始创建网络。这里的actor_critic是一个函数输入，该函数会调用core.MLPActorCritic。则actor_critic=core.MLPActorCritic，所以说ac=core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs) 
    ac_targ = deepcopy(ac) # 定义目标网络
    # 到此网络完成创建！

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        # ！！这种一般就是固定用法，如果使用了目标网络，目标网络是不需要更新参数的，它的参数都是从Q网络按照比例复制而来的。
        # 那么根据前面说的，如果使用了nn.modular，却有确定不需要计算梯度，将梯度置为False可以避免而外的torch自动求导带来的存储上的占用。
        # nn.Module 类提供了 .parameters() 方法，该方法用于返回模型中包含的所有可训练参数的生成器（Generator）。
        # 这个生成器可以通过迭代访问模型中的所有可训练参数。
        
    # List of parameters for both Q-networks (save this for convenience) <------
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    # ！！！重点知识——迭代器：ac.q1.parameters()本身就是一个迭代器，换句话说你可以使用for param in ac.q1.parameters(): print(param)
    # 遍历神经网络中的所有parameters，而这里的itertools.chain函数的作用是把两个迭代器粘在一起，变成一个连续的迭代器。

    # Experience buffer，初始化replay buffer。
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!) 统计整个网络中所有参数的总数
    # core.count_vars(module)是自定义的函数，返回module网络中参数的总数。
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    # 从Logger class来看，这个函数的作用是print一行字。

    # Set up function for computing SAC Q-losses
    # 到了Q loss的重点地方了。搞明白（1）如何计算loss的（2）怎么把loss回传的
    def compute_loss_q(data):
        # data是replaybuffer中使用sample batch函数拿出来的，还记得当时replaybuffer返回的是一个torch tensor的字典
        # 还需要的注意的是，到了使用replay buffer的地方，都是使用的batch，replay buffer中就没有定义只使用一个transition的函数。
        # 使用batch就要注意，输入都将会是多条数据。
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)
        # 这两个双Q网络是分别初始化，同步训练的，他们之间的网络参数没有任何共享，所以只要用到计算Q(s,a)的地方，就一定会同时出现q1，q2.
        # ac是一个MLPActorCritic class，ac.q1,ac.q2,ac.pi都是在MLPActorCritic class中定义的self.q1（就是一个网络）

        # Bellman backup for Q functions
        # 还记得之前说的，一但出现只使用target network的时候，就可以使用该命令，避免浪费计算资源。
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values，Q(s,a)网络的输出是一个标量值
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # 注意，这个地方是SAC的soft bellman backup。参考SAC原文的（2）（3）。此外，原文这个地方是没有weight alpha的版本，alpha是一个控制
            # 熵正则项大小的超参，可以使用自适应技巧动态调节。同时这个地方就能看出来，当时求log_pi的时候有一个.sum(axis=1)，将action所有维度都求和，
            # 的理解是正确的，因为Q函数的输出是一个标量，在大多数情况下是无法与action space对齐的，所以要想做这个地方的q_pi_targ - logp_a2运算，
            # 一定是把logpi(a|s)所有维度对应的log probability求和了。
            # 虽然前面使用了batch，q_pi_targ不是一个标量，而是一个batch大小的标量，但是这里的理解是没有问题的，只要假设batch的大小为1就可以。
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        # 这个地方求mean是batch更新导致的，而且很常见，求最终loss的时候肯定是多个数据导出loss的均值。
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2 
        # 这里使用两个Q网络梯度的平均来更新网络是值得深思的。直觉上这里可能会带来无法收敛，不稳定的问题，但是实际使用下来效果却恰恰相反
        # 使用梯度平均是一种平滑梯度的策略，目的是提高算法的稳定性，并且在实践中表现得相当有效
        # ！！！所以说，RL有很多很多都是经验上的结论，其数学原理并不一定能被解释清楚，这些东西就需要你看这些大牛写的代码，然后不断积累了。
        
        # Useful info for logging
        # 首先这是个字典，然后创建字典的时候，键值对是用“=”连在一起的，虽然存完后变成了“：”但是写的时候用的是“=”
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())
        # Q值被通过.detach().numpy()的操作从PyTorch张量中分离出来，并转换为NumPy数组。.detach()的作用是创建一个新的张量，与原始张量共享相同的底层数据，但不再追踪梯度。.numpy()则将PyTorch张量转换为NumPy数组。

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        # 你会发现这里面有些计算在算loss_q的时候计算过，我觉得有以下原因：（1）不要让loss_pi loss_q混起来，宁愿让同样的数据过两遍NN
        # （2）在走main loop的时候，不一定是各自都更新一次，所以不要把两个loss写成一个function。
        o = data['obs'] # pi的loss是-Q+正则项，这个值越小越好
        pi, logp_pi = ac.pi(o) 
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss，J(\theta) = -q + alpha*logp_pi, 值得注意的是SAC中在进行bellman更新的时候，就是柔性bellman backup，
        # 这个和你的算法有着显著的区别，所以这个地方得到的Q函数，本身就是soft Q(s,a)。
        loss_pi = (alpha * logp_pi - q_pi).mean() # logp_pi对于一个transition已经是一个标量了，mean是对batch求的。

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info
        # 到此两个loss就都确定完了，下面就准备更新了。同时注意到，上面定义的两个loss函数都是sac函数里面的函数，所以说当前还是有一个主流程的
        # 这也是为什么下面会有未封装的代码块的原因，等sac全部结束还要理一遍数据在sac函数中怎么流动的。

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    # 这个地方定义优化器函数，Adam的好处是它自适应地调节每个参数的学习率（是的，即便你给的是确定的学习率），adam的输入是参数和学习率，具体见文档。
    # 硬要说代码逻辑，其实前面就可以把pi网络的参数ac.pi.parameters()保存下来了，只不过前面只存了q_params。

    # Set up model saving
    logger.setup_pytorch_saver(ac) # 这个是存放模型，虽然之后这种东西基本不会修改，那个logger文件，你也得看掉，。

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        # PyTorch中，梯度默认是会累积的，而在每次优化步骤之前需要清零，否则会影响梯度下降的计算。你可能会想，不清零会怎么样
        # 首先数学的角度，每个loss算出来的梯度，就在当前的更新步骤中使用，没有看到需要累积的情况。
        # 其次，如果不清空，梯度会一直加起来，而不是覆盖，首先会导致梯度方向有问题，其次可能会导致梯度爆炸。
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # loss.backward(): 计算梯度。
        # q_optimizer.step(): 使用优化器根据计算得到的梯度来更新Q网络的参数。
        # 换句话说，使用backward只是计算出了梯度，因为梯度是确定的，但是如何更新有很多技巧（例如学习率），这都是优化器决定的。
        # 因此让优化器走一步，才会更新网络参数。至于优化器会更新哪个网络的参数，是在定义优化器的时候就确定好的：q_optimizer = Adam(q_params, lr=lr)

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort computing gradients for them during the policy learning step.
        # 这里代码原来就解释的很清楚了，冻住的目的不一定是为了部分更新等高级操作，很可能就是为了节省算力。
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
                # .data 是 PyTorch 中 Tensor 的一个属性，它提供了对底层数据的直接访问，但不会记录梯度信息。.data 的使用在 PyTorch 0.4 版本及之前比较常见。在更新后的版本（0.4 之后），PyTorch 引入了tensor.detach()函数来达到类似的效果，同时更加安全。
                # .mul_() 和 .add_() 是 PyTorch Tensor 类中的 in-place 操作函数。这些函数在现有的 Tensor 上执行操作，而不是创建新的 Tensor。
                # tensor.mul_(value)：对 Tensor 中的每个元素都乘以给定的值 value，并将结果保存在原始的 Tensor 中。这是一个 in-place 的操作。
                # tensor.add_(value)：对 Tensor 中的每个元素都加上给定的值 value，并将结果保存在原始的 Tensor 中。同样，这也是一个 in-place 操作。

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
        # 这个地方你可能会想，为什么前面obs直接就扔到了网络里面，而这个地方却要加一个将o转换为torch tensor的操作呢？
        # ANS：因为前面操作都是从replay-buffer中拿的数据，而replay buffer最后的输出已经将所有obs都转换为tensor了，
        # 其实也可以想想，在进行神经网络运算的时候，一定是用tensor安全，怎么可能直接把observation丢进去。
        # 而这个地方，由于是get action，很有可能是与环境交互，得到的obs是环境直接返回的，因此在这里要加上一个将o转换为torch tensor的操作。    

    # 这个test函数是这样，它没有输入，只要执行test_agent()命令，就会用当前策略玩num_test_episodes把游戏，并且把每局游戏的累计奖励写进log文件
    # 这也解释了，为什么对于Mujoco这种环境不会终止的游戏，奖励最终也不会无限大（智能体最多就进行比如说2000步交互，如果策略不再提升，奖励也就平稳了）
    def test_agent():
        for j in range(num_test_episodes):
            # 这些函数都是SAC的内部函数，test_env变量在调用SAC的时候就会输入。同时这个地方也能弄明白两点：
            # （1）与环境交互是单个state单个动作进行的，无法使用tensor，因为gym API并不支持从任意state开始
            # （2）解释了为什么上面get_action函数需要将obs转换为tensor；以及在什么情况下会使用deterministic policy。
            # （3）换句话说，我们的代码这个，地方是不是也要考虑使用确定性策略？
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            # 这个地方应该是不论测试几把，反正都先记录下来，然后再log文件中计算奖励的均值与方差，最后存放最大，最小，均值，方差几个值，
            # 所以说具体存什么还是得看log

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs # 很重要，关系到总interaction，一般我们所谓的1millio就是与环境交互的次数。
    start_time = time.time() # 这个是记录系统时间
    o, ep_ret, ep_len = env.reset(), 0, 0 
    # 初始化环境，by the way，其实完全可以把这些SAC算法中会执行的命令放在一块，而不是夹在内部函数之中，spinningup这么做主要还是为了理解。

    # Main loop: collect experience in env and update/log each epoch
    # 这个for循环会直接让智能体与环境交互1million次（total_steps = steps_per_epoch * epochs），再交互过程中网络就会按照一定频率更新，
    # 也就是说，只要运行一次SAC函数，就相当于完成了一轮完整的训练，log文件中就存放了能做出完成performance图的全部数据。
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()
        # 第一步：先得到一个动作

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # 第二步：与环境进行一次交互，环境进入下一状态

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d # <--- ??? 这个地方不知道是在handle哪种特殊情况，但反正正常情况下影响不大。

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d) # replay buffer 会把单个的输入存成list的形式。

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2 # 这一步确实很重要，当然你也可以每次让算法都返回一个当前环境，比如这里让o=env.state，但是并不是所有的gym env都支持这个API。

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
        # 到达done则记录累计回报，然后重置环境。

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)
        # 这个地方有点奇怪，假设update_every=50，这个for循环会让神经网络进行50次更新？虽然这样做没什么数学上的问题，但是有必要这样做吗？<---???

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            # 如果step达到了4000次的倍数，则epoch+1。

            # Save model
            # save freq默认为1，也就是每个epoch结束后都存一次，epoch达到最大后一定存一次。
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            # 上面定义的test_agent函数是没有任何返回的，它会把测试的结果（当前策略打一局游戏的累积奖励）直接写进log文件

            # Log info about epoch
            # 到此SAC算法运行一次，把该存的存下来。
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
