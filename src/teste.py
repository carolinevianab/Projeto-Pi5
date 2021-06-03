from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np
import random 
from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
from matplotlib import pyplot as plt

# class QNetwork():
#     def __init__(self, state_dim, action_size):


#         # self.input = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.states, name='input')
#         # self.q_true = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
#         # self.a_true = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
#         # self.reward = tf.placeholder(dtype=tf.float32, shape=[], name='reward')



#         self.state_in = tf.placeholder(dtype=tf.float32, shape=(None, ) + state_dim)
#         self.action_in = tf.placeholder(tf.int32, shape=[None])
#         self.q_target_in = tf.placeholder(tf.float32, shape=[None])
#         action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
#         self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
#         self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
#         self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
        
#         self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

#     def update_model(self, session, state, action, q_target):
#         feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
#         session.run(self.optimizer, feed_dict=feed)
    
#     def get_q_state(self, session, state):
#         q_state = session.run(self.q_state, feed_dict={self.state_in: state})
#         return q_state

# class ReplayBuffer():
#     def __init__(self, maxlen):
#         self.buffer = deque(maxlen=maxlen)
        
#     def add(self, experience):
#         self.buffer.append(experience)
        
#     def sample(self, batch_size):
#         sample_size = min(len(self.buffer), batch_size)
#         samples = random.choices(self.buffer, k=sample_size)
#         return map(list, zip(*samples))

# class DQNAgent():
#     def __init__(self, env):
#         self.state_dim = env.observation_space.shape
#         self.action_size = env.action_space.n
#         self.q_network = QNetwork(self.state_dim, self.action_size)
#         self.replay_buffer = ReplayBuffer(maxlen=10000)
#         self.gamma = 0.97
#         self.eps = 1.0
        
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
        
#     def get_action(self, state):
#         q_state = self.q_network.get_q_state(self.sess, [state])
#         action_greedy = np.argmax(q_state)
#         action_random = np.random.randint(self.action_size)
#         action = action_random if random.random() < self.eps else action_greedy
#         return action
    
#     def train(self, state, action, next_state, reward, done):
#         self.replay_buffer.add((state, action, next_state, reward, done))
#         states, actions, next_states, rewards, dones = self.replay_buffer.sample(50)
#         q_next_states = self.q_network.get_q_state(self.sess, next_states)
#         q_next_states[dones] = np.zeros([self.action_size])
#         q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)
#         self.q_network.update_model(self.sess, states, actions, q_targets)
        
#         if done: self.eps = max(0.1, 0.99*self.eps)
    
#     def __del__(self):
#         self.sess.close()



class DQNAgent:
    """ DQN agent """
    def __init__(self, states, actions, max_memory, double_q):
        self.states = states
        self.actions = actions
        self.session = tf.Session()
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=10)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.memory = deque(maxlen=max_memory)
        self.eps = 1
        self.eps_decay = 0.99999975
        self.eps_min = 0.1
        self.gamma = 0.90
        self.batch_size = 32
        self.burnin = 100000
        self.copy = 10000
        self.step = 0
        self.learn_each = 3
        self.learn_step = 0
        self.save_each = 500000
        self.double_q = double_q
    def build_model(self):
            """ Model builder function """
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.states, name='input')
            self.q_true = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
            self.a_true = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.reward = tf.placeholder(dtype=tf.float32, shape=[], name='reward')
            self.input_float = tf.to_float(self.input) / 255.
            # Online network
            with tf.variable_scope('online'):
                self.conv_1 = tf.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                self.conv_2 = tf.layers.conv2d(inputs=self.conv_1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                self.conv_3 = tf.layers.conv2d(inputs=self.conv_2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                self.flatten = tf.layers.flatten(inputs=self.conv_3)
                self.dense = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu)
                self.output = tf.layers.dense(inputs=self.dense, units=self.actions, name='output')
            # Target network
            with tf.variable_scope('target'):
                self.conv_1_target = tf.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                self.conv_2_target = tf.layers.conv2d(inputs=self.conv_1_target, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                self.conv_3_target = tf.layers.conv2d(inputs=self.conv_2_target, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                self.flatten_target = tf.layers.flatten(inputs=self.conv_3_target)
                self.dense_target = tf.layers.dense(inputs=self.flatten_target, units=512, activation=tf.nn.relu)
                self.output_target = tf.stop_gradient(tf.layers.dense(inputs=self.dense_target, units=self.actions, name='output_target'))
            # Optimizer
            self.action = tf.argmax(input=self.output, axis=1)
            self.q_pred = tf.gather_nd(params=self.output, indices=tf.stack([tf.range(tf.shape(self.a_true)[0]), self.a_true], axis=1))
            self.loss = tf.losses.huber_loss(labels=self.q_true, predictions=self.q_pred)
            self.train = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(self.loss)
            # Summaries
            self.summaries = tf.summary.merge([
                tf.summary.scalar('reward', self.reward),
                tf.summary.scalar('loss', self.loss),
                tf.summary.scalar('max_q', tf.reduce_max(self.output))
            ])
            self.writer = tf.summary.FileWriter(logdir='./logs', graph=self.session.graph)
    def copy_model(self):
            """ Copy weights to target network """
            self.session.run([tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('target'), tf.trainable_variables('online'))])
    def save_model(self):
            """ Saves current model to disk """
            self.saver.save(sess=self.session, save_path='./models/model', global_step=self.step)
    def add(self, experience):
            """ Add observation to experience """
            self.memory.append(experience)
    def predict(self, model, state):
            """ Prediction """
            if model == 'online':
                return self.session.run(fetches=self.output, feed_dict={self.input: state})
            if model == 'target':
                return self.session.run(fetches=self.output_target, feed_dict={self.input: np.array(state)})
    def run(self, state):
            """ Perform action """
            if np.random.rand() < self.eps:
                # Random action
                action = np.random.randint(low=0, high=self.actions)
            else:
                # Policy action
                q = self.predict('online', np.expand_dims(state, 0))
                action = np.argmax(q)
            # Decrease eps
            self.eps *= self.eps_decay
            self.eps = max(self.eps_min, self.eps)
            # Increment step
            self.step += 1
            return action
    def learn(self):
            """ Gradient descent """
            # Sync target network
            if self.step % self.copy == 0:
                self.copy_model()
            # Checkpoint model
            if self.step % self.save_each == 0:
                self.save_model()
            # Break if burn-in
            if self.step < self.burnin:
                return
            # Break if no training
            if self.learn_step < self.learn_each:
                self.learn_step += 1
                return
            # Sample batch
            batch = random.sample(self.memory, self.batch_size)
            state, next_state, action, reward, done = map(np.array, zip(*batch))
            # Get next q values from target network
            next_q = self.predict('target', next_state)
            # Calculate discounted future reward
            if self.double_q:
                q = self.predict('online', next_state)
                a = np.argmax(q, axis=1)
                target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
            else:
                target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1)
            # Update model
            summary, _ = self.session.run(fetches=[self.summaries, self.train],
                                        feed_dict={self.input: state,
                                                    self.q_true: np.array(target_q),
                                                    self.a_true: np.array(action),
                                                    self.reward: np.mean(reward)})
            # Reset learn step
            self.learn_step = 0
            # Write
            self.writer.add_summary(summary, self.step)

    


#main
env = gym.make('Contra-v0')
env = JoypadSpace(env, RIGHT_ONLY)
actions = (84, 84, 4)

print("actions", env.action_space)
print("observation_space ", env.observation_space.shape[0])

agent = DQNAgent(actions, env.action_space.n, max_memory=100000, double_q=True)

for ep in range(1000):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.run(state=state)
        next_state, reward, done, info = env.step(action)
        agent.add(experience=(state, next_state, action, reward, done))
        # Update agent
        agent.learn()
        env.render()
        total_reward += reward
        state = next_state
        
    print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))