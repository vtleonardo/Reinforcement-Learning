import gym
import os
import numpy as np
import random
import pandas as pd
from ReplayMemory import ReplayMemory

import tensorflow as tf
seed=1
random.seed(seed)
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)
#Silenciando as mensagens iniciais do tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import Model, load_model
from keras.layers import Conv2D, Flatten, Dense, Lambda, Input, multiply
from keras import backend as K
from keras.optimizers import RMSprop, Adam
import time
import dqn_wrappers
import sys
import imageio
path_save = "../DQN2/Imagens"

"""
Classe que cria um agente que utilizará a arquitetura DQN para o aprendizado em um ambiente.
Atributos Recebidos:
    env (gym environment) = ambiente de aprendizado do agente. 
                            Default: Game Breakout v4 (Consultar documentação da gym para mais detalhes).
    num_episodes (int) = número total de episódios do aprendizado.
                            Default: 500 episódios.
    discount_rate (float) = discount rate gamma.
                            Default: discount rate de 0.99.
    lr (float) = learning rate da rede neural.
                            Default: learning rate de 0.001.
    epsilon (float) = variável da probabilidade de exploração com seu valor inicial (política e-greedy).
                            Default: epsilon 1.0 (100% de exploração no começo da aprendizagem).
    e_min (float) = variável com o valor final da probabilidade de exploração (após o decaimento).
                            Default: epsilon 0.1 (10% de exploração ao final do decaimento).
    e_exep_decay (int) = taxa do decaimento exponencial (quanto maior mais lento 1/e.decay).
                            Default: decaimento exponencial de 50 (decaimento de 63.2% em 50 episódios).
    e_lin_decay (int) = taxa do decaimento linear (número de episódios para chegar ao valor de e_min).
                            Default: decaimento linear de 10000.
    linear_decay_mode (bool) = Habilita o decaimento linear ou desabilita(levando a escolha do decaimento exponencial).
                            Default: False (Decaimento exponencial habilitado).
    target_update (int) = número de STEPS no qual os parâmetros de Q_hat serão atualizados.
                            Default: atualização a cada 2000.
"""
class AgentDQN:
    def __init__(self,
                 env=gym.make('BreakoutDeterministic-v4'),
                 num_train_frames=10000000,
                 discount_rate=0.99,
                 lr = 0.00025,
                 epsilon=1.0,
                 e_min=0.1,
                 e_exp_decay=50,
                 e_lin_decay=1000000,
                 linear_decay_mode=False,
                 target_update = 10000,
                 num_states_stored = 1000000,
                 batch_size = 32,
                 input_size = (84,84),
                 history_size = 4,
                 num_random_play = 50000,
                 load=False,
                 path_save_plot = "PlotDQN",
                 path_save_weights = "WeightsDQN"
                 ):
        self.env = dqn_wrappers.wrap_deepmind(env)
        self.num_train_frames = num_train_frames
        #Aux variável para contar o número total de steps
        self.steps_cont = 0
        #Número de ações possíveis e formato dos estados de entrada
        self.actions_num = 3#env.action_space.n
        self.input_size = input_size
        #Definindo as dimensões de entrada da network
        self.input_shape = self.input_size+(history_size,)

        #=========Parâmetros de aprendizagem===========#
        self.discount_rate = discount_rate
        self.lr = lr
        self.epsilon = epsilon
        self.e_min = e_min
        self.e_exp_decay = e_exp_decay
        self.e_lin_decay = e_lin_decay
        self.linear_decay_mode = linear_decay_mode
        self.target_update = target_update
        self.Q_value = self.initalize_network("Q_value",load)
        self.Q_hat = self.initalize_network("Q_hat")
        self.update_Q_hat()

        self.error_clip = 1.0

        # Inicializando a memória de replay
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(num_states_stored,batch_size)
        self.history_size = history_size
        self.num_random_play = num_random_play
        #======Variáveis auxiliares para controle ou plot=======#
        self.start_episode = 0
        self.i_episode = 0
        self.loss_value = 0.0
        self.q_rate = 0.0
        self.values_dict = {"Rewards":[],"Loss":[],"Q_value":[],"Num_frames":[], "Tempo":[]}
        self.image_array=[]
        self.reward_100 = 0

        self.path_save_plot = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_save_plot)
        self.path_save_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_save_weights)
        self.initialize_tensor()



    """
    Função que cria as redes neurais
    Atributos Recebidos:
        name = nome da rede neural
    """
    def initalize_network(self, name,load=False):
        frames_input = Input(self.input_shape, name=name)
        actions_input = Input((self.actions_num,), name='filter')
        lamb = Lambda(lambda x: (2 * x - 255) / 255.0, )(frames_input)
        conv_1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(lamb)
        conv_2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv_1)
        conv_3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv_2)
        conv_flattened = Flatten()(conv_3)
        hidden = Dense(512, activation='relu')(conv_flattened)
        output = Dense(self.actions_num)(hidden)
        filtered_output = multiply([output,actions_input])
        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        self.sess = tf.Session()
        K.set_session(self.sess)
        if(load):
            print("Carregando Modelo")
            model.load_weights(os.path.join(self.path_save_weights,"meu_weight-tensor-{}.h5".format(self.env.spec.id)))
        model.summary()
        return model

    def initialize_tensor(self):
        self.state = tf.placeholder(tf.uint8, [None] + list(self.input_shape))
        self.action = tf.placeholder(tf.int32, [None])
        self.reward = tf.placeholder(tf.float32, [None])
        self.state_next= tf.placeholder(tf.uint8, [None] + list(self.input_shape))
        self.done = tf.placeholder(tf.float32, [None])
        state_float = tf.cast(self.state, tf.float32)
        state_next_float = tf.cast(self.state_next, tf.float32)

        act_one_hot = tf.one_hot(self.action, self.actions_num, on_value=1.0, off_value=0.0)
        self.mask_one = tf.ones_like(act_one_hot, tf.float32)
        current_q = tf.reduce_sum(self.Q_value([state_float, act_one_hot]),axis=1)
        prediction = self.Q_hat([state_next_float, self.mask_one])
        target_q = tf.reduce_max(prediction, axis=1)
        target_val = tf.stop_gradient(self.reward + (self.discount_rate * target_q) * (1 - self.done))
        #self.loss_train = tf.nn.l2_loss(target_val - current_q) / self.batch_size
        difference = tf.abs(current_q - target_val)
        quadratic_part = tf.clip_by_value(difference, 0.0, self.error_clip)
        linear_part = difference - quadratic_part
        error = (0.5 * tf.square(quadratic_part)) + (self.error_clip * linear_part)
        self.loss_train = tf.reduce_mean(error)
        #self.train_op = tf.train.RMSPropOptimizer(
        #     self.lr, decay=0.95, momentum=0.0, epsilon=0.01).minimize(self.loss_train)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_train)
        self.sess.run(tf.global_variables_initializer())

    """
    Função que atualiza os parâmetros de Q_hat iguais aos de Q_value
    """
    def update_Q_hat(self):
        self.Q_hat.set_weights(self.Q_value.get_weights())

    """
    Função que seleciona a ação com base na política e-greedy.
        Atributos Recebidos:
        state = conjunto de estados com formato input_shape que serão utilizados para selecionar a ação.
    """
    def e_greddy_action(self, state):
        # Seleciona uma ação aleatoriamente
        if np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.actions_num))
        # Seleciona a ação com o maior Q-value
        else:
            prediction = self.Q_value.predict_on_batch([state,np.ones((1,self.actions_num))])
            self.q_rate += np.amax(prediction)
            action = np.argmax(prediction)
        return action

    """
    Função que realiza o decaimento de epsilon linearmente ou exponencialmente, saturando no valor mínimo.
    """
    def decay_epsilon(self):
        if self.linear_decay_mode:
            self.epsilon = np.amax((self.e_min, -((1.0-self.e_min)* self.steps_cont)/self.e_lin_decay + 1.0))
        else:
            self.epsilon = np.amin((1, (self.e_min + (1.0-self.e_min) * np.exp(-(self.i_episode-1) / self.e_exp_decay))))

    """
    Função que atualiza o histórico, retirando o primeiro elemento, movendo os demais e acrescentando
    o estado novo ao final do histórico de estados.
        Atributos Recebidos:
        history = historico dos estados com formato input_shape.
        state_next = estado mais rescente que será adionado ao final do histórico.
    """
    def refresh_history(self,history,state_next):
        history[:,:,:-1] = history[:,:,1:]
        history[:,:,-1] = state_next[:,:,0]
        return history

    def train_dqn(self):
        #Amostrando uniformemente estados da memória e em sequência realizando o unpack
        st,act,r,st_next,d,idx = self.replay_memory.sample()
        self.loss += self.sess.run([self.train_op, self.loss_train],
                             feed_dict={self.state: st, self.action: act, self.reward: r,
                                        self.state_next: st_next, self.done: d})[1]

    """
    Função que verifica se a pasta onde os arquivos serão salvos existe. Se não ele a cria.
    """
    def folder_exists(self,path_save):
        if not (os.path.exists(path_save)):
            os.mkdir(path_save)

    def save_agent(self):
        self.folder_exists(self.path_save_weights)
        self.folder_exists(self.path_save_plot)
        self.Q_value.save_weights(os.path.join(self.path_save_weights,
                                               "meu_weight-tensor-{}.h5".format(self.env.spec.id)))
        df = pd.DataFrame.from_dict(self.values_dict)
        self.reward_100=df["Rewards"].tail(100).mean()
        df.to_csv(os.path.join(self.path_save_plot,'valores-tensor-{}.csv'.format(self.env.spec.id)),index=False)

    def run(self, DEBUG=True, aleatorio = False, to_render= False, save_gif = False):
        self.steps_cont=0
        time_it = time.time()
        self.i_episode = 0
        #aux=0
        while self.steps_cont < self.num_train_frames:
            self.i_episode += 1
            #Enchendo a replay memory
            if(aleatorio and self.replay_memory.size()>=self.num_random_play):
                break
            state = self.env.reset()
            state = np.concatenate((state,state,state,state),axis=2)
            #======Inicializando variáveis====#
            done = False
            t = 0
            self.loss = 0
            self.q_rate = 0
            reward_total_episode = 0
            avg_loss = 0
            avg_q_rate = 0
            while not done:
                if (to_render and not aleatorio):self.env.render()
                t += 1
                self.steps_cont += 1
                action = self.e_greddy_action(state.reshape((1,) + state.shape))
                if(action>0):
                    action_real = action + 1
                else:
                    action_real = action
                state_next, reward, done, _ = self.env.step(action_real)
                if done and self.env.unwrapped.ale.lives():
                    self.env.reset()
                    done=False
                state_next = self.refresh_history(np.copy(state), state_next)
                self.replay_memory.append(state,action,reward,state_next,done)
                state = np.copy(state_next)
                if not aleatorio:
                    reward_total_episode += reward
                    self.train_dqn()
                    self.decay_epsilon()
                    if (self.steps_cont % self.target_update == 0):
                        if(DEBUG):print("Renovei Q_hat")
                        self.save_agent()
                        self.update_Q_hat()

            if not aleatorio:
                avg_loss = self.loss / (t + 1)
                avg_q_rate = self.q_rate / (t + 1)
                self.values_dict["Rewards"].append(reward_total_episode)
                self.values_dict["Loss"].append(avg_loss)
                self.values_dict["Q_value"].append(avg_q_rate)
                self.values_dict["Num_frames"].append(self.steps_cont)
                self.values_dict["Tempo"].append(time.time()-time_it)

            if (DEBUG):
                fps= t/(time.time()-time_it)
                print("Episódio {:d}, Total de Frames:{:d}/{:d}, Time steps:{:d} , Reward total:{:.2f}, "
                      "Reward médio {:d} frames:{:.3f}"
                      " Epsilon:{:.4f}, Tamanho da memória:{:d}, Loss:{:.4f}, Q-value médio:{:.4f}, FPS:{:.2f}, "
                      "Tempo(s):{:.3f}".format(self.i_episode,
                    self.steps_cont,self.num_train_frames, t, reward_total_episode, self.target_update, self.reward_100,
                    self.epsilon, self.replay_memory.size(),avg_loss,avg_q_rate,fps,time.time()-time_it))
                time_it = time.time()


if __name__ == "__main__":
    dqn = AgentDQN(env=gym.make('PongNoFrameskip-v4'),linear_decay_mode=True, load=False,lr=1e-4)
    print(dqn.env.spec.id)
    dqn.env.seed(seed)
    dqn.run(aleatorio=True,DEBUG=True)
    dqn.run(to_render=False,DEBUG=True)