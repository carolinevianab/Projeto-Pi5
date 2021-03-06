# --- Responsavel pela inicialização do agente, pela sua rede neural DQN, e seu treinamento --- #

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import random
import numpy as np
from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from matplotlib import pyplot as plt

class DQNAgent:
    """ DQN agent """
    # faz a inicialização de todas as coisas necessarias para treinamento, como: as ações do agente, a montagem da rede neural ...
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
        # Online network -> rede neural principal.
        with tf.variable_scope('online'):
            self.conv_1 = tf.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            self.conv_2 = tf.layers.conv2d(inputs=self.conv_1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            self.conv_3 = tf.layers.conv2d(inputs=self.conv_2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
            self.flatten = tf.layers.flatten(inputs=self.conv_3)
            self.dense = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu)
            self.output = tf.layers.dense(inputs=self.dense, units=self.actions, name='output')
        # Target network -> rede neural usada para setar os "Q_values"(pontuação) para as ações. essa rede neural ajuda no output da rede neural principal.
        with tf.variable_scope('target'):
            self.conv_1_target = tf.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            self.conv_2_target = tf.layers.conv2d(inputs=self.conv_1_target, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            self.conv_3_target = tf.layers.conv2d(inputs=self.conv_2_target, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
            self.flatten_target = tf.layers.flatten(inputs=self.conv_3_target)
            self.dense_target = tf.layers.dense(inputs=self.flatten_target, units=512, activation=tf.nn.relu)
            self.output_target = tf.stop_gradient(tf.layers.dense(inputs=self.dense_target, units=self.actions, name='output_target'))
        # Optimizer -> contem as informações para fazer o treinamento do modelo.
        self.action = tf.argmax(input=self.output, axis=1)
        self.q_pred = tf.gather_nd(params=self.output, indices=tf.stack([tf.range(tf.shape(self.a_true)[0]), self.a_true], axis=1))
        self.loss = tf.losses.huber_loss(labels=self.q_true, predictions=self.q_pred)
        self.train = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(self.loss)
        # Summaries -> contem um resumo com as principais informações da session do TF
        self.summaries = tf.summary.merge([
            tf.summary.scalar('reward', self.reward),
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('max_q', tf.reduce_max(self.output))
        ])
        self.writer = tf.summary.FileWriter(logdir='./logs', graph=self.session.graph)

    def copy_model(self):
        """ Copy weights to target network """ #-> copia os pesos da rede neural principal, para a target network
        self.session.run([tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('target'), tf.trainable_variables('online'))])

    def save_model(self):
        """ Saves current model to disk """
        self.saver.save(sess=self.session, save_path='./models/model', global_step=self.step)

    def add(self, experience):
        """ Add observation to experience """ #-> adiciona a observação atual do agente no ambiente na sua experiencia.
        self.memory.append(experience)

    def predict(self, model, state):
        """ Prediction """ #-> com a session do TF, tenta adivinhar qual a melhor ação para o estado atual.
        if model == 'online':
            return self.session.run(fetches=self.output, feed_dict={self.input: np.array(state)})
        if model == 'target':
            return self.session.run(fetches=self.output_target, feed_dict={self.input: np.array(state)})

    def run(self, state):
        """ Perform action """
        if np.random.rand() < self.eps: #epsilon -> diz o quão o agente é provavel em tomar uma ação aleatória, oq ajuda no aprendizado.
            # Random action
            action = np.random.randint(low=0, high=self.actions)
        else:
            # Policy action
            q = self.predict('online', np.expand_dims(state, 0)) # retorna as ações com as melhores pontuações
            action = np.argmax(q) # -> ação com o maior valor de q
        # Decrease eps
        self.eps *= self.eps_decay # vai diminuindo o epsilon para forçar o agente a tomar ações aleatórias.
        self.esp = max(self.eps_min, self.eps) # não deixa ser menor q o limite.
        # Increment step
        self.step += 1
        return action

    def learn(self):
        """ Gradient descent """
        
        ## Responsavel pelo treinamento

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
        batch = random.sample(self.memory, self.batch_size) # -> lote(pedaço) da mémoria da rede neural
        state, next_state, action, reward, done = map(np.array, zip(*batch)) # -> retorna os valores para suas devidas variaveis
        # Get next q values from target network
        next_q = self.predict('target', next_state) # -> usa a rede neural 'target' que vai prever o q_value
        # Calculate discounted future reward
        if self.double_q: # -> double_q -> trabalha com duas tabelas de Q_Value, auxiliando no treinamento
            q = self.predict('online', next_state)
            a = np.argmax(q, axis=1)
            target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a] # faz o calculo(bellman) do "q_value"(pontuação) para a ação.
        else:
            target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1) # same as up comment #
        # Update model -> atualiza o sumario do modelo.
        summary, _ = self.session.run(fetches=[self.summaries, self.train],
                                      feed_dict={self.input: state,
                                                 self.q_true: np.array(target_q),
                                                 self.a_true: np.array(action),
                                                 self.reward: np.mean(reward)})
        # Reset learn step
        self.learn_step = 0
        # Write
        self.writer.add_summary(summary, self.step)

    def replay(self, env, model_path, n_replay):
        """ Model replay """
        ckpt = tf.train.latest_checkpoint(model_path)
        saver = tf.train.import_meta_graph(ckpt + '.meta')
        graph = tf.get_default_graph()
        input_t = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('online/output/BiasAdd:0')
        # Replay RL agent
        total_reward = 0
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            for _ in range(n_replay):
                state = env.reset()
                while True:
                    time.sleep(0.01)
                    env.render()
                    if np.random.rand() < 0.0:
                        action = np.random.randint(low=0, high=self.actions, size=1)[0]
                    else:
                        q = sess.run(fetches=output, feed_dict={input_t: np.expand_dims(state, 0)})
                        action = np.argmax(q)
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    state = next_state
                    if done:
                        os.system("cls")
                        print("Score do agente: ", info['score'])
                        input("... Aperte qualquer tecla para continuar ...")
                        break
        env.close()