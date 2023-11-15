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
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs) # 到此网络完成创建
    ac_targ = deepcopy(ac) # 定义目标网络

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        # ！！这种一般就是固定用法，如果使用了目标网络，目标网络是不需要更新参数的，它的参数都是从Q网络按照比例复制而来的。
        # 那么根据前面说的，如果使用了nn.modular，却有确定不需要计算梯度，将梯度置为False可以避免而外的torch自动求导带来的存储上的占用。
        # nn.Module 类提供了 .parameters() 方法，该方法用于返回模型中包含的所有可训练参数的生成器（Generator）。
        # 这个生成器可以通过迭代访问模型中的所有可训练参数。
        
    # List of parameters for both Q-networks (save this for convenience) <------
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
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

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
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
