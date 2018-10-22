################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################
import os
import numpy as np
import pandas as pd
from ReplayMemory import ReplayMemory
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Lambda, Input, multiply
from keras import backend as K
import time
import random
from Environments import WrapperGym#, WrapperDoom
import tensorflow as tf
from tensorflow import set_random_seed
from utils import printd, folder_exists
import utils
import imageio

#Setting the DEBUG lvl of the function printd (utils.py)
utils.DEBUG = True
utils.DEBUG_lvl = 1

#Setting all random seeds to a value known
seed=1
np.random.seed(seed)
random.seed(seed)
set_random_seed(seed)
#Silencing tensorflow
if utils.DEBUG_lvl <= 2:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AgentDQN:
    """
    Class that create the DQN agent that will interact with an environment.
    """

    def __init__(self,
                 env='PongNoFrameskip-v4',
                 num_simul_frames=10000000,
                 discount_rate=0.99,
                 lr=0.00025,
                 epsilon=1.0,
                 e_min=0.1,
                 e_lin_decay=1000000,
                 linear_decay_mode=True,
                 e_exp_decay=300,
                 target_update=10000,
                 num_states_stored=1000000,
                 batch_size=32,
                 input_size=(84,84),
                 history_size=4,
                 num_random_play=50000,
                 load=False,
                 steps_save_weights = 50000,
                 steps_save_plot=10000,
                 save_episodes_flag=False,
                 steps_save_episodes=50,
                 path_save_episodes="Episodes",
                 weights_load_path="",
                 loss_type="huber",
                 optimizer="rmsprop",
                 path_save_plot="Plot",
                 path_save_weights="Weights",
                 silent_mode = False,
                 multi_gpu = False,
                 gpu_device = "0"
                 ):
        """

        :param  env : str (Default : PongNoFrameskip-v4 (atari gym environment [see gym documentation for more
                details])
                    The name of the environment where the agent will interact.

        :param  num_simul_frames : int (Default : 10 000 000)
                    Total number of frames that the agent will be trained.

        :param  discount_rate : float (Default: 0.99)
                    Discount rate gamma (RL parameter).

        :param  lr : float (Default: 0.00025)
                    Neural Network learning rate.

        :param  epsilon : float (Default: 1.0 (100% of exploration))
                    Initial value of random actions (exploration) chosen by the agent using the policy e-greedy.

        :param  e_min : float (Default: 0.1 (10% of exploration))
                    Final value of random actions (exploration) chosen by the agent using the policy e-greedy.

        :param  e_lin_decay : int (Default: 1 000 000)
                    Number of frames for epsilon reach its final value linearly (e_min).

        :param  linear_decay_mode : bool (Default: True - linear decay enable).
                    Enables the linear decay mode of epsilon or disable(thus enabling the exponential
                    decay mode).

        :param  e_exp_decay : int (Default:300 [ie 63.2% of decay in 300 episodes])
                    Exponential decay rate in EPISODES (Bigger values slowly is the decay
                    since exp^[-1/e.decay]).

        :param  target_update : int (Default:10 000)
                    Number of frames that the parameters of Q_hat will be update with the parameters of Q_value
                    [See the DQN paper for more details].

        :param  num_states_stored : int (Default: 1 000 000)
                    Number of states stored in the replay memory.

        :param  batch_size : int (Default: 32)
                    Number of samples uniformly sampled from the replay memory to do the Neural Network
                    (NN) training.

        :param  input_size : tuple (int) (Default: (84,84))
                    Width x Height of the input frame (image) that will be send to the Neural Network.

        :param  history_size : int (Default: 4)
                    Number of sequential frames that will be stacked together to form the input volume
                    to the NN.

        :param  num_random_play : int (Default: 50 000)
                    Number of states generate by actions chosen randomly that will be stored in the
                    replay memory before the DQN algorithm trains begin.

        :param  load : bool (Default: False)
                    Variable that controls if it's to load the weights from a external .h5 file generated
                    by another simulation.

        :param  steps_save_weights : int (Default: 50 000)
                    Desired number of frames to save the weights.

        :param  steps_save_plot : int (Default: 1 000)
                    Desired number of frames to save the plot variables.

        :param  save_episodes_flag : bool (Default: False)
                    Flag that controls if it's to save episodes on the disk.

        :param  steps_save_episodes : int (Default: 50)
                    Each N (steps_save_episodes) episodes an episode will be saved on the disk as .gif.

        :param  path_save_episodes : str (Default: "Episodes")
                    Path to the folder where will be saved the episode as .gif file.

        :param  weights_load_path : str (Default: "")
                    Path of the .h5 file with the weights of the Network to be loaded.

        :param  loss_type : str (Default: "hubber")
                    Name of the type of loss function that will be used to train the DQN Network. There are two
                    possible types: "huber" and "MSE".

        :param  optimizer : str (Default: "rmsprop")
                    Name of the type of optimizer that will be used to train the DQN Network. There are two
                    possible types: "rmsprop" and "adam". The first one uses the setting described on the
                    DQN paper, the second uses the tensorflow/keras default parameters.

        :param  path_save_plot : str (Default: "Plot")
                    Folder's path where will be saved the .csv file with the algorithm's information.

        :param  path_save_plot : str (Default: "Weights")
                    Folder's path where will be saved the .h5 file with the Neural Network Weights.

        :param  silent_mode : bool (Default : False)
                    If it's active no message will be displayed on the prompt.

        :param  multi_gpu : bool (Default : False)
                    If false, you can select what gpu to use (default: first gpu).

        :param gpu_device : str (Default : "0")
                    String with the number of the gpu device that will be used in the case of the multi_gpu
                    variable is False.
        """
        #Setting the silent mode
        if silent_mode:
            utils.DEBUG = False

        self.multi_gpu = multi_gpu
        self.gpu_device = gpu_device
        if not self.multi_gpu:
            # The GPU id to use
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_device

        if "doom" in env.lower():
            pass
            #self.env = WrapperDoom.WrapperDoom(env)
        else:
            self.env = WrapperGym.WrapperGym(env)

        #Total number of frames that the simulation will run
        self.num_simul_frames = num_simul_frames
        #Counter to the total number of steps
        self.steps_cont = 0
        #Total number of actions possible inside the environment
        self.actions_num = self.env.numberOfActions()
        #Size(Width x Height) of the input images
        self.input_size = input_size
        #Defining the input of the network (Input volume shape)
        self.input_shape = self.input_size+(history_size,)
        self.load = load
        self.weights_load_path = weights_load_path
        #=========Learning Parameters===========#
        self.discount_rate = discount_rate
        self.lr = lr
        self.epsilon = epsilon
        self.e_min = e_min
        self.e_exp_decay = e_exp_decay
        self.e_lin_decay = e_lin_decay
        self.linear_decay_mode = linear_decay_mode
        self.target_update = target_update
        self.Q_value = self.initalize_network("Q_value")
        self.Q_hat = self.initalize_network("Q_hat")
        self.update_Q_hat()
        self.loss_type = loss_type
        self.optimizer = optimizer

        #Clipping the error between the interval of 0.0 and 1.0 to compute the Huber Loss
        self.error_clip = 1.0

        #Inicializing the Replay memory
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(num_states_stored=num_states_stored,batch_size=batch_size)
        self.history_size = history_size
        #Number of random plays to fill the replay memory before the RL-algorithm begins
        self.num_random_play = num_random_play

        #======Auxiliary variables to plot or control=======#
        self.start_episode = 0
        self.i_episode = 0
        self.loss_value = 0.0
        self.q_rate = 0.0
        self.values_dict = {"Rewards":[],"Loss":[],"Q_value":[],"Num_frames":[],
                                                                            "Time":[], "FPS":[], "Epsion":[]}
        self.image_array=[]
        self.reward_100 = 0

        self.steps_save_weights = steps_save_weights
        self.steps_save_plot = steps_save_plot
        self.save_episodes_flag = save_episodes_flag
        self.steps_save_episodes = steps_save_episodes
        # Checking if the default paths exists.
        if path_save_episodes == "Episodes":
            self.path_save_episodes=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                                    path_save_episodes)
            folder_exists(self.path_save_episodes)
        if path_save_plot == "Plot":
            self.path_save_plot=os.path.join(os.path.dirname(os.path.realpath(__file__)), path_save_plot)
            folder_exists(self.path_save_plot)
        if path_save_weights == "Weights":
            self.path_save_weights=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                                    path_save_weights)
            folder_exists(self.path_save_weights)

        # Initializing the graph an its variables.
        self.initialize_graph()
        # Creating a log file
        self.LOG_FILENAME = os.path.join(self.path_save_plot, 'Training-{}.txt'.format(self.env.getName()))
        # Just opening and cleaning previous files from another simulations
        with open(self.LOG_FILENAME, "w") as text_file:
            pass
        self.summary()


    def summary(self):
        """
        Function that display this object information.

        :return:
            nothing
        """
        strr = ""
        strr += "\n============================================================================================"
        strr +="\nINITIALIZING THE DQN ALGORITHM WITH THE FOLLOWING SETTINGS:"
        strr +="\n\tEnvironment: {}".format(self.env.getName())
        strr +="\n\tTotal number of frames to be be simulated: {} frame(s)".format(self.num_simul_frames)
        strr +="\n\tDiscount rate: {}".format(self.discount_rate)
        strr +="\n\tInitial Epsilon: {}".format(self.epsilon)
        strr +="\n\tFinal Epsilon: {}".format(self.e_min)
        if self.linear_decay_mode:
            strr +="\n\tLinear Decay mode is activated!"
            strr +="\n\tThe final Epsilon will be reached in: {} frame(s)".format(self.e_lin_decay)
        else:
            strr +="\n\tExponential Decay mode is activated!"
            strr +="\n\tThe final Epsilon will be reached in approximately: {} episode(s)"\
                                                                                .format(self.e_exp_decay*5)
        strr +="\n\tLearning rate: {}".format(self.lr)
        strr +="\n\tBatch size: {}".format(self.batch_size)
        strr +="\n\tState shape: {}".format(self.input_shape)
        strr +="\n\tThe Network will have the {} loss".format(self.loss_type.upper())
        strr +="\n\tThe Network will be trained using {} optimizer".format(self.optimizer.upper())
        strr +="\n\tThe Target Network will be updated every: {} frame(s)".format(self.target_update)
        strr +="\n\tThe Replay Memory will store: {} state(s)".format(self.replay_memory.num_states_stored)
        strr +="\n\tApproximate number of states from random plays before training: {} state(s)"\
                                                                                .format(self.num_random_play)
        if self.save_episodes_flag:
            strr += "\n\tThe episode will be saved in: {}".format(self.path_save_episodes)
            strr += "\n\tAn episode will be saved each {} episodes".format(self.steps_save_episodes)
        strr +="\n\tThe information will be saved in: {}".format(self.path_save_plot)
        strr +="\n\tThe plot variables will be saved each: {} frame(s)".format(self.steps_save_plot)
        strr +="\n\tThe NN weights will be saved in: {}".format(self.path_save_weights)
        if self.load:
            strr +="\n\tLoad the weights is set to True!"
            strr +="\n\tThe weights {} will be loaded from the {} folder ".format(self.weights_load_path,
                                                                                       self.path_save_weights)
        strr +="\n\tThe weights will be saved each: {} frame(s)".format(self.steps_save_weights)
        strr +="\n\tMulti gpu mode : {}".format(self.multi_gpu)
        if self.multi_gpu:
            strr +="\n\tGPU device used : {}".format(self.gpu_device)
        strr +="\n============================================================================================="
        printd(strr)
        with open(self.LOG_FILENAME, "a+") as text_file:
            print(strr, file=text_file)


    def initalize_network(self, name):
        """
        Function that creates the neural network and the Tensorflow session.

        This function creates the Neural Network (NN) with the parameters given by the DQN paper (Nature) using
        Keras framework together with Tensorflow. Before the Input volume enters in the network, its pixels are
        normalized (see lambda function on the code for more details). The network receives 2 inputs, the first
        is the image volume, and the last is a "Mask/Filter". The Mask has the function of filter the desired Q
        values and clear the rest. Besides that the weights can be loaded from an external .h5 file, the option
        to load "load" and the path to this file are given on initialization.

        :param  name : str
                The NN's name

        :return nothing
        
        """
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
        if(self.load):
            if self.weights_load_path != "":
                printd("Loading the Model {}!".format(self.weights_load_path))
                model.load_weights(self.weights_load_path)
            else:
                printd("Load option is True but no path to a valid .h5 file was given"
                       " therefore no weights was loadaded!")
        if (utils.DEBUG and utils.DEBUG_lvl >=2):
            model.summary()
        return model


    def initialize_graph(self):
        """
        Function that initializes the tensorflow graph that computes the NN training.

        This function creates the tensorflow graph that will compute the NN error and training. The network
        can be trained by two different losses, hubber loss (default) and mse (Mean Squared Error), the type of
        loss used is defined on the initialization of this class by the parameter "loss_type". This function
        allows you to choose between two optimizers: RMSProp (with the settings given by the DQN Paper) and
        ADAM (with tensorflow defaults arguments), this choice is also made on the initialization by the 
        parameter "optimizer".
        OBS: The tensorflow graph is fed by dictionary in the function train_dqn().

        :param  nothing

        :return nothing

        """
        #Defining the tensors variables (placeholds)
        self.state = tf.placeholder(tf.uint8, [None] + list(self.input_shape))
        self.action = tf.placeholder(tf.int32, [None])
        self.reward = tf.placeholder(tf.float32, [None])
        self.state_next= tf.placeholder(tf.uint8, [None] + list(self.input_shape))
        self.done = tf.placeholder(tf.float32, [None])
        #Defining the operations
        state_float = tf.cast(self.state, tf.float32)
        state_next_float = tf.cast(self.state_next, tf.float32)
        act_one_hot = tf.one_hot(self.action, self.actions_num, on_value=1.0, off_value=0.0)
        self.mask_one = tf.ones_like(act_one_hot, tf.float32)
        current_q = tf.reduce_sum(self.Q_value([state_float, act_one_hot]),axis=1)
        prediction = self.Q_hat([state_next_float, self.mask_one])
        target_q = tf.reduce_max(prediction, axis=1)
        #Computing the NN ERROR as descrived in the DQN paper.
        target_val = tf.stop_gradient(self.reward + (self.discount_rate * target_q) * (1 - self.done))
        if "huber" in self.loss_type:
            # Computing the Huber Loss
            self.loss_train = tf.losses.huber_loss(current_q,target_val)
        elif "mse" in self.loss_type:
            #Computing the MSE loss
            self.loss_train = tf.losses.mean_squared_error(target_val,current_q)
        if "rms" in self.optimizer.lower():
            #Using RMSprop with DQN paper's parameters
            self.train_op = tf.train.RMSPropOptimizer(
                 self.lr, decay=0.95, momentum=0.0, epsilon=0.01).minimize(self.loss_train)
        elif "adam" in self.optimizer.lower():
            #Using the Adam  optimizer with default parameters
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_train)
        #Initializing the variables
        self.sess.run(tf.global_variables_initializer())


    def update_Q_hat(self):
        """
        Function that updates the values of Q_hat with the values of Q_value at each N (:param target_update)
        steps.
        
        :param  nothing
        :return nothing
        
        """
        self.Q_hat.set_weights(self.Q_value.get_weights())


    def e_greddy_action(self, state):
        """
        Function that selects an action with base on the e-greedy police.

        :param  state : input volume (np.array) of shape input_shape (dtype=np.int8)
                    A volume compound of a set of states (images) of depth "history_size".

        :return nothing
        """

        # Gets a random action if the variable self.epsilon is less than a random variable
        # (distributed between 0 and 1)
        if np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.actions_num))
        # Otherwise the algorithm computes the Q value of each action possible for that state and
        # pick the one with greater value.
        else:
            prediction = self.Q_value.predict_on_batch([state,np.ones((1,self.actions_num))])
            self.q_rate += np.amax(prediction)
            action = np.argmax(prediction)
        return action


    def decay_epsilon(self):
        """
        Function that makes the epsilon decay. This decay can be linear if the initialization parameter
        linear_decay_mode is True(default) or exponential otherwise. This function doesn't receives any
        parameter, it only uses the instance parameters (self.):
            e_min : float
                Minimum value of epsilon
            e_lin_decay: int
                Number of frames that the function will reach its minimum
            steps_cont: int
                Current number of frames
            i_episode: int
                Current number of episodes
                
        :param  nothing

        :return nothing

        """
        if self.linear_decay_mode:
            # straight line equation wrapper by max operation -> max(min_value,(-mx + b))
            self.epsilon = np.amax((self.e_min, -((1.0-self.e_min)* self.steps_cont)/self.e_lin_decay + 1.0))
        else:
            # exponential's function Const(e^-t) wrapped by a min function
            self.epsilon = np.amin((1, (self.e_min + (1.0-self.e_min) * np.exp(-(self.i_episode-1) /
                                                                                        self.e_exp_decay))))


    def train_dqn(self):
        """
        Function that trains the NN. It uniformly samples a sample from the replay memory. The replay memory
        returns a set composed of: state,action,reward, state_next, done (if state_next is terminal),
        idx (iterator). These variables are used to feed the tensorflow graph that computes the loss
        and updates de weights.
        
        :param  nothing

        :return nothing

        """
        # Uniform sampling from the replay memory
        st,act,r,st_next,d,idx = self.replay_memory.sample()
        self.loss += self.sess.run([self.train_op, self.loss_train],
                             feed_dict={self.state: st, self.action: act, self.reward: r,
                                        self.state_next: st_next, self.done: d})[1]

    def save_episode(self,saved_episode):
        """
        Function that saves the episode as .gif file. An episode is saved each N number of episodes
        (defined by steps_save_episodes on initialization).

        :param  saved_episode : np.array (dtype=np.uint8).
                    Sequence of frames concatenated in a np.array (dtype=np.uint8).

        :return nothing

        """
        imageio.mimsave(os.path.join(self.path_save_episodes, "Episode-{}.gif".format(self.i_episode)),
                        np.rollaxis(saved_episode, 2, 0))

    def save_weights(self):
        """
        Function that saves the weights in .h5 file. The weights are saved each N steps
        (defined by steps_save_weights on initialization).
        
        :param  nothing

        :return nothing

        """
        self.Q_value.save_weights(os.path.join(self.path_save_weights,
                                    "weights-{}-{}.h5".format(self.env.getName(),self.steps_cont)))


    def save_plot(self):
        """
        Function that saves the variables to plot in a .csv file. The variables
        are saved each N steps (defined by steps_save_plot on initialization).

        :param  nothing

        :return nothing

        """
        df = pd.DataFrame.from_dict(self.values_dict)
        self.reward_100 = df["Rewards"].tail(10).mean()
        df.to_csv(os.path.join(self.path_save_plot, '-{}.csv'.format(self.env.getName())), index=False)

    def refresh_history(self, history, state_next):
            """
            Function that updates the history (a set of "n" frames that is used as a state of the replay memory)
            taking out the first frame, moving the rest and adding the new frame to end of the history.

            :param history : input volume of shape input_shape
                    The history that will be refreshed (basically a set of n frames concatenated
                    [np.array dtype=np.int8]) as a state on the replay memory.

            :param state_next : Image (np.array of dtype=np.uint8)
                    Frame (np.array dtype=np.int8) of the environment current state after a action was take.

            :return nothing
            """
            history[:, :, :-1] = history[:, :, 1:]
            history[:, :, -1] = state_next[:, :, 0]
            return history

    def run(self, random_fill = False, to_render= False):
        """
        Function that runs the RL-DQN algorithm as demonstrated in the DQN paper.
        
        :param  random_fill : bool (default False)
                Variable that decides if it is to fill the replay memory with states that come of 
                random actions.If false the DQN algorithm will run otherwise it'll only choose random action
                e get the correspondent states from the environment to fill the replay memory.

        :param  to_render : bool (default False)
                Variable that decides if it's to render on the screen the current episode.
                OBS: If this variable is true the fps will decrease harshly since it needs to
                show the game in reasonable speed.

        :return nothing

        """
        self.steps_cont=0
        time_it = time.time()
        self.i_episode = 0
        saved_episode = 0
        while self.steps_cont < self.num_simul_frames:
            self.i_episode += 1
            # Filling the replay memory until it reaches the number num_random_play
            if(random_fill and self.replay_memory.size()>=self.num_random_play):
                break
            state = self.env.reset()
            if self.save_episodes_flag: saved_episode = state
            #Transforming the receive state (image frame) in a volume of n frames (history)
            state = np.concatenate((state,state,state,state),axis=2)
            #======Initializing variables====#
            done = False
            t = 0
            self.loss = 0
            self.q_rate = 0
            reward_total_episode = 0
            avg_loss = 0
            avg_q_rate = 0
            while not done:
                if (to_render and not random_fill):self.env.render()
                #the variable "t" differs from steps_cont on that it is reseted on each loop end
                t += 1
                #accomulate the total number of frames
                self.steps_cont += 1
                action = self.e_greddy_action(state.reshape((1,) + state.shape))
                state_next, reward, done, _ = self.env.step(action)
                if self.save_episodes_flag: saved_episode = np.concatenate((saved_episode,state_next),axis=2)
                #Transforming the state_next (image frame) in a volume of n frames (history)
                state_next = self.refresh_history(np.copy(state), state_next)
                self.replay_memory.append(state,action,reward,state_next,done)
                state = np.copy(state_next)
                reward_total_episode += reward
                if not random_fill and (self.replay_memory.size() > self.batch_size):
                    self.train_dqn()
                    self.decay_epsilon()
                    if (self.steps_cont % self.target_update == 0):
                        printd("Q_hat was renewed!",lvl=2)
                        self.update_Q_hat()
                    if (self.steps_cont % self.steps_save_weights == 0):
                        self.save_weights()
                    if (self.steps_cont % self.steps_save_plot == 0):
                        self.save_plot()
            
            
            if self.save_episodes_flag and not random_fill and(self.i_episode % self.steps_save_episodes == 0):
                self.save_episode(saved_episode)
                
            fps = t / (time.time() - time_it)
            if not random_fill:
                avg_loss = self.loss / (t + 1)
                avg_q_rate = self.q_rate / (t + 1)
                self.values_dict["Rewards"].append(reward_total_episode)
                self.values_dict["Loss"].append(avg_loss)
                self.values_dict["Q_value"].append(avg_q_rate)
                self.values_dict["Num_frames"].append(self.steps_cont)
                self.values_dict["Time"].append(time.time()-time_it)
                self.values_dict["FPS"].append(fps)
                self.values_dict["Epsion"].append(self.epsilon)

            strr = ""
            strr+="Episode {:d}:".format(self.i_episode)
            strr+="\n\t\t\tTotal Frames: {:d}/{:d},".format(self.steps_cont,self.num_simul_frames,)
            strr+="\n\t\t\tFrames in this episode: {:d},".format(t)
            strr+="\n\t\t\tTotal reward: {:.2f},".format(reward_total_episode)
            strr+="\n\t\t\tMean reward of 10 episodes: {:.3f}".format(self.reward_100)
            strr+="\n\t\t\tEpsilon: {:.4f},".format(self.epsilon)
            strr+="\n\t\t\tReplay memory size: {:d}/{:d},".format(self.replay_memory.size(),
                                                                self.replay_memory.num_states_stored)
            strr+="\n\t\t\tLoss: {:.4f},".format(avg_loss)
            strr+="\n\t\t\tMean Q value: {:.4f},".format(avg_q_rate)
            strr+="\n\t\t\tFPS: {:.2f}, ".format(fps)
            strr+="\n\t\t\tTime of this episode: {:.3f} (s)".format(time.time()-time_it)
            with open(self.LOG_FILENAME, "a+") as text_file:
                print(strr, file=text_file)
            printd(strr)
            time_it = time.time()


if __name__ == "__main__":
    dqn = AgentDQN(env='PongNoFrameskip-v4', lr=1e-4,optimizer="adam",num_states_stored=100000,
                   num_random_play=10000,e_min=0.02,e_lin_decay=100000,target_update=1000,
                   save_episodes_flag=True,steps_save_episodes=10)
    dqn.env.set_seed(seed)
    printd("EXECUTING RANDOM PLAYS TO FILL THE REPLAY MEMORY")
    dqn.run(random_fill=True)
    printd("EXECUTING AND TRAINING DQN ALGORITHM")
    dqn.run(to_render=False)