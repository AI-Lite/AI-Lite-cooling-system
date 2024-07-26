import argparse
import os
import time

# import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tensorflow
import tensorflow_probability as tfp
# import tensorlfow.compat.v1.layers as tl
import tensorlayer as tl
# tf.disable_eager_execution()

# tensorflow.random.set_seed(1)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 1  # random seed
RENDER = False  # render while training

ALG_NAME = 'PPO'
# TRAIN_EPISODES = 1000  # total number of episodes for training
# TEST_EPISODES = 10  # total number of episodes for testing
# MAX_STEPS = 200  # total number of steps for each episode
GAMMA = 0.9  # reward discount
# LR_A = 0.005 # learning rate for actor
# LR_C = 0.01  # learning rate for critic
LR_A = 0.005 # learning rate for actor
LR_C = 0.01  # learning rate for critic
BATCH_SIZE = 8  # update batch size
# ACTOR_UPDATE_STEPS = 60  # actor update steps
# CRITIC_UPDATE_STEPS = 60  # critic update steps
ACTOR_UPDATE_STEPS = 40  # actor update steps
CRITIC_UPDATE_STEPS = 50  # critic update steps

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.18


###############################  PPO  ####################################


class PPO(object):
    """
    PPO class
    """
    def __init__(self, state_dim, Num_hidden_1, action_dim, method='clip'):
        # critic
        with tf.name_scope('critic'):
            # inputs = tf.keras.Input(shape=[state_dim, ], batch_size=None)
            # layer = tf.keras.layers.Dense(Num_hidden_1, tf.nn.relu)(inputs)
            # # layer = tl.layers.Dense(64, tf.nn.relu)(layer)
            # v = tf.keras.layers.Dense(1)(layer)
            inputs = tl.layers.Input([None,state_dim])
            layer = tl.layers.Dense(int(Num_hidden_1), tf.nn.relu)(inputs)
            # layer = tl.layers.Dense(int(Num_hidden_1*0.5), tf.nn.relu)(layer)
            v = tl.layers.Dense(1)(layer)
        # self.critic = tf.keras.Model(inputs, v)
        self.critic = tl.models.Model(inputs, v)
        self.critic.train()

        # actor
        with tf.name_scope('actor'):
            # # inputs = tf.keras.Input([None, state_dim], tf.float32, 'state')
            # inputs = tf.keras.Input(shape=[state_dim, ])
            # layer = tf.keras.layers.Dense(Num_hidden_1, tf.nn.relu)(inputs)
            # # layer = tl.layers.Dense(64, tf.nn.relu)(layer)
            # a = tf.keras.layers.Dense(action_dim, tf.nn.softmax)(layer)
            inputs = tl.layers.Input([None,state_dim])
            layer = tl.layers.Dense(Num_hidden_1, tf.nn.sigmoid)(inputs)
            layer = tl.layers.Dense(Num_hidden_1, tf.nn.sigmoid)(layer)
            a = tl.layers.Dense(action_dim, tf.nn.softmax)(layer)
            # act_prob = tf.nn.softmax(a, name='act_prob')  # 加一个softmax回归，将动作值转化为概率值
            # prob = tl.layers.Lambda(lambda x: x*2, name='lambda')(a)
            # logstd = tf.Variable(np.zeros(action_dim, dtype=np.float32))
        self.actor = tl.models.Model(inputs, a)
        # self.actor = tl.models.Model(inputs, mean)
        # self.actor.trainable_weights.append(logstd)
        # self.actor.logstd = logstd
        self.actor.train()

        self.actor_opt = tf.keras.optimizers.Adam(LR_A)
        self.critic_opt = tf.keras.optimizers.Adam(LR_C)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        # self.action_bound = action_bound

    def train_actor(self, state, action, adv, pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        with tf.GradientTape() as tape:
            # mean, std = self.actor(state), tf.exp(self.actor.logstd)
            # pi = tfp.distributions.Normal(mean, std)

            act_prob = self.actor(state)
            #print(act_prob)
            # print(act_prob[0])

            # print((action))
            # new_prob = act_prob[action]
            # print(new_prob)
            new_prob = tf.convert_to_tensor([act_prob[i][0][element] for i, element in enumerate(action)])
            old_prob = tf.convert_to_tensor([pi[i][0][element] for i, element in enumerate(action)])
            # print(new_prob)
            # print(old_prob)
            # act_prob = tf.nn.softmax(act_prob, name='act_prob')
            ratio = new_prob/old_prob
            # print(ratio)
            # print(act_prob(action))
            # print
            # print(adv)
            surr = ratio * adv
            if self.method == 'penalty':  # ppo penalty
                kl = tfp.distributions.kl_divergence(pi.numpy(), act_prob.numpy())
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:  # ppo clip
                loss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
                )
        a_gard = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))
        return loss

        if self.method == 'kl_pen':
            return kl_mean

    def train_critic(self, reward, state):
        """
        Update actor network
        :param reward: cumulative reward batch
        :param state: state batch
        :return: None
        """
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.int32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        # mean, std = self.actor(s), tf.exp(self.actor.logstd)
        # pi = tfp.distributions.Normal(mean, std)
        pi = self.actor(s)
        adv = r - self.critic(s)

        # update actor
        if self.method == 'kl_pen':
            for _ in range(ACTOR_UPDATE_STEPS):
                kl = self.train_actor(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            print("actor training begins:##########")
            for _ in range(ACTOR_UPDATE_STEPS):
                loss = self.train_actor(s, a, adv, pi)
                print(loss)

        # update critic
        for _ in range(CRITIC_UPDATE_STEPS):
            self.train_critic(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def get_action(self, state):
        """
        Choose action
        :param state: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        # state = state[np.newaxis, :].astype(np.float32)
        # print(state)
        # mean, std = self.actor(state), tf.exp(self.actor.logstd)
        # print(state)
        act_prob = self.actor(state)
        # self.sess1 = tf.Session()
        # print(act_prob)
        # act_prob = tf.nn.softmax(act_prob, name='act_prob')
        # with tf.Session() as sess:
        act_action = act_prob.numpy()
        return act_action
        # if greedy:
        #     action = mean[0]
        # else:
        #     pi = tfp.distributions.Normal(mean, std)
        #     action = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        # return np.clip(action, -self.action_bound, self.action_bound)

    # def save(self):
    #     """
    #     save trained weights
    #     :return: None
    #     """
    #     path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
    #     tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
    #
    # def load(self):
    #     """
    #     load trained weights
    #     :return: None
    #     """
    #     path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
    #     tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
    #     tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_batch(self, next_state):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        v_s_ = self.critic(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

    def perceive(self, state, action, reward, next_state):
        self.store_transition(state, action, reward)
        if len(self.state_buffer) >= BATCH_SIZE:
            self.finish_batch(next_state)
            self.update()


if __name__ == '__main__':
    state_dim = 10
    num_hidden = 2
    action_dim = 6
    agent = PPO(state_dim, num_hidden, action_dim)

    state = [[[1.,         0.04280335, 0.82,       0.,         1.,         1.0117753, 1.0702102,  0.,        1.,   0.5916119]]]
    action = agent.get_action(state)
    # buffer=[]
    # state1 = [1.0,2.0,3.0]
    # buffer.append(state1)
    # print(buffer)
    # action = agent.get_action([buffer])
    # print(action)
    # buffer.append(state1)
    # print(buffer)
    # action = agent.get_action([buffer])
    # print(action)
    # # print(action)

    # env = gym.make(ENV_ID).unwrapped
    #
    # # reproducible
    # env.seed(RANDOM_SEED)
    # np.random.seed(RANDOM_SEED)
    # tf.random.set_seed(RANDOM_SEED)
    #
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # action_bound = env.action_space.high
    #
    # agent = PPO(state_dim, action_dim, action_bound)
    #
    # t0 = time.time()
    # if args.train:
    #     all_episode_reward = []
    #     for episode in range(TRAIN_EPISODES):
    #         state = env.reset()
    #         episode_reward = 0
    #         for step in range(MAX_STEPS):  # in one episode
    #             if RENDER:
    #                 env.render()
    #             action = agent.get_action(state)
    #             state_, reward, done, info = env.step(action)
    #             agent.store_transition(state, action, reward)
    #             state = state_
    #             episode_reward += reward
    #
    #             # update ppo
    #             if len(agent.state_buffer) >= BATCH_SIZE:
    #                 agent.finish_batch(state_)
    #                 agent.update()
    #         agent.finish_batch(state_, done)
    #         print(
    #             'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
    #                 episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0)
    #         )
    #         if episode == 0:
    #             all_episode_reward.append(episode_reward)
    #         else:
    #             all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
    #     agent.save()
    #
    #     plt.plot(all_episode_reward)
    #     if not os.path.exists('image'):
    #         os.makedirs('image')
    #     plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
    #
    # if args.test:
    #     # test
    #     agent.load()
    #     for episode in range(TEST_EPISODES):
    #         state = env.reset()
    #         episode_reward = 0
    #         for step in range(MAX_STEPS):
    #             env.render()
    #             state, reward, done, info = env.step(agent.get_action(state, greedy=True))
    #             episode_reward += reward
    #             if done:
    #                 break
    #         print(
    #             'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
    #                 episode + 1, TEST_EPISODES, episode_reward,
    #                 time.time() - t0))