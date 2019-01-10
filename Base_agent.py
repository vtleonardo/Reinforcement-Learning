################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################
import os
import numpy as np
import pandas as pd
from ReplayMemory import ReplayMemory
from keras import backend as K
import time
import random
from Environments import WrapperGym, WrapperDoom
import tensorflow as tf
from tensorflow import set_random_seed
from utils import printd, folder_exists, str2bool, read_cfg
import utils
import imageio
import argparse
import threading
import re
import Networks
import sys

# Multi-thread lock
lock = threading.Lock()

# Setting the DEBUG lvl of the function printd (utils.py)
utils.DEBUG = True
utils.DEBUG_lvl = 1

# Silencing tensorflow
if utils.DEBUG_lvl <= 2:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Agent:
    """
    Class that creates the agent that will interact with an environment.
    """

    def __init__(self,
                 agent_name="DQN",
                 mode="train",
                 env='PongNoFrameskip-v4',
                 include_score=False,
                 config_file_path="DoomScenarios/labyrinth.cfg",
                 network_model="DQN",
                 normalize_input=True,
                 frame_skip=4,
                 num_simul_frames=10000000,
                 discount_rate=0.99,
                 lr=0.00025,
                 epsilon=1.0,
                 e_min=0.1,
                 decay_mode="linear",
                 e_lin_decay=1000000,
                 e_exp_decay=300,
                 target_update=10000,
                 num_states_stored=1000000,
                 batch_size=32,
                 input_shape=(84, 84, 1),
                 history_size=4,
                 num_random_play=50000,
                 load_weights=False,
                 steps_save_weights=50000,
                 steps_save_plot=10000,
                 to_save_episodes=False,
                 steps_save_episodes=50,
                 path_save_episodes="Episodes",
                 weights_load_path="",
                 loss_type="huber",
                 optimizer="rmsprop",
                 path_save_plot="Plot",
                 path_save_weights="Weights",
                 silent_mode=False,
                 multi_gpu=False,
                 gpu_device="0",
                 multi_threading=False,
                 is_recurrent=False
                 ):
        """
        :param  agent_name : str (Default : "DQN")
                    Agent's name, it will be passed to the saved files (weights, episodes, plot).

        :param mode : str (Default : train)
                    Execution mode of the algorithm. There are two possible modes: "train" and "test".
                    The first one trains the DQN/DRQN agent, training the Neural Netwok (NN) with the
                    experiences, in an environment, the second one tests it without storing the experiences and
                    training the NN.

        :param  env : str (Default : PongNoFrameskip-v4 (atari gym environment [see gym documentation for more
                details])
                    The name of the environment where the agent will interact.

        :param  include_score: [GYM ATARI EXCLUSIVE] bool (Default: False)
                    If its to include in the state image the score from the environment (atari game).

        :param config_file_path: [DOOM EXCLUSIVE] str (path) (Default : "DoomScenarios/labyrinth.cfg")
                    Path to .cfg file that contains the configuration to the Doom's environment.

        :param  network_model : str (Default : DQN (Same architecture used in Nature's DQN paper )])
                    Neural Network's architecture to be used. The name should match
                    one of the methods inside the Networks.py. You can create you own model inside the
                    Networks.py and send the name of the method to this argument to be implemented by the agent.

        :param  normalize_input : bool (Default: True)
                    Variable that controls if it's to normalize the state's pixels or not.

        :param  frame_skip : int (Default : 4)
                    Total number of frames that will be skipped between states.

        :param  num_simul_frames : int (Default : 10 000 000)
                    Total number of frames that the agent will be trained.

        :param  discount_rate : float (Default: 0.99)
                    Discount rate gamma (RL parameter).

        :param  lr : float (Default: 0.00025)
                    Neural Network's learning rate.

        :param  epsilon : float (Default: 1.0 (100% of exploration))
                    Probability's initial value of the agent to choose random actions (exploration) using the
                    policy e-greedy.

        :param  e_min : float (Default: 0.1 (10% of exploration))
                    Probability's final value of the agent to choose random actions (exploration) using the
                    policy e-greedy.

        :param  decay_mode : str (Default: linear - linear decay enable).
                    Type of epsilon's decay mode. There are two possible types: "linear" and "exponential".

        :param  e_lin_decay : int (Default: 1 000 000)
                    Number of frames for epsilon to reach its final value linearly (e_min).

        :param  e_exp_decay : int (Default:300 [ie 63.2% of decay in 300 episodes])
                    Exponential decay rate in EPISODES (The decay is slowly with bigger values since the
                    decay equation is exp^[-1/e_exp_decay]).

        :param  target_update : int (Default:10 000)
                    Number of frames that the parameters of Q_target will be updated with the parameters of Q.
                    [See the DQN paper for more details].

        :param  num_states_stored : int (Default: 1 000 000)
                    Number of states stored in the replay memory.

        :param  batch_size : int (Default: 32)
                    The batch's size to train the Neural Network.

        :param  input_shape : tuple (int) (Default: (84,84))
                    Input frame's shape (WxHxColor_channel[if any]) that will be sent to the Neural Network.
                    If just WxH are entered, the color_channel will be 1 (gray_scale)

        :param  history_size : int (Default: 4)
                    Number of sequential frames that will be stacked together to form the input volume
                    to the NN.

        :param  num_random_play : int (Default: 50 000)
                    Number of states generated by actions chosen randomly that will be stored in the
                    replay memory before the agent's training begins.

        :param  load_weights : bool (Default: False)
                    Variable that controls if it's to load the weights from a external .h5 file generated
                    by another simulation.

        :param  steps_save_weights : int (Default: 50 000)
                    Desired number of frames to save the weights.

        :param  steps_save_plot : int (Default: 1 000)
                    Desired number of frames to save the plot variables.

        :param  to_save_episodes : bool (Default: False)
                    Flag that controls if it's to save episodes on the disk.

        :param  steps_save_episodes : int (Default: 50)
                    Number of episodes that an episode will be saved on the disk as .gif.

        :param  path_save_episodes : str (Default: "Episodes")
                    Path to the folder where will be saved the episode as .gif file.

        :param  weights_load_path : str (Default: "")
                    Path of the .h5 file with the weights of the Network to be loaded.

        :param  loss_type : str (Default: "huber")
                    Name of the type of loss function that will be used to train the Network. There are two
                    possible types: "huber" and "MSE".

        :param  optimizer : str (Default: "rmsprop")
                    Name of the type of optimizer that will be used to train the Network. There are two
                    possible types: "rmsprop" and "adam". The first one uses the setting described on the
                    DQN paper, the second uses the tensorflow/keras default parameters.

        :param  path_save_plot : str (Default: "Plot")
                    Folder's path where will be saved the .csv file with the algorithm's information.

        :param  path_save_plot : str (Default: "Weights")
                    Folder's path where will be saved the .h5 file with the Neural Network Weights.

        :param  silent_mode : bool (Default : False)
                    If it's active no message will be displayed on the prompt (The logging keeps active).

        :param  multi_gpu : bool (Default : False)
                    If false, you can select what gpu to use (if there is more than one).

        :param  gpu_device : int (Default : 0 [first gpu])
                    The ID of the gpu device that will be used in the case of the multi_gpu variable
                    is False. To use the gpu, send -1.

        :param  multi_threading : bool (Default : False)
                    If this mode is active the sampling part of the algorithm will be done in parallel with
                    the main RL-algorithm. Therefore, when we call the train method the sample from the
                    replay memory will be ready (cutting the execution time of the algorithm thus increasing
                    the fps). The main drawback of this approach is that we insert a delay of one time step in
                    the algorithm, in other words, the experience of a time step t can only be sampled at time
                    t+1. However, tests show that is no perceptible effect on the
                    learning.
        :param  multi_gpu : bool (Default : False)
                    If false, you can select what gpu to use (if there is more than one).

        :param  is_recurrent : bool (Default : False)
                    If your model has any recurrent layer set this flag to True.
        """
        # Setting the mode
        self.mode = mode
        # Setting the root path
        self.root_path = os.path.dirname(os.path.realpath(__file__))
        # Defining the agent's name
        self.agent_name = agent_name
        # Setting the silent mode
        if silent_mode:
            utils.DEBUG = False

        self.multi_gpu = multi_gpu
        self.gpu_device = gpu_device
        self.multi_threading = multi_threading
        if not self.multi_gpu:
            # The GPU id to use
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu_device)

        # Adding the third dimension, case the input received is only compose of Width x Height
        if len(input_shape) == 2:
            input_shape = input_shape + (1,)
        # Changing from notation (x,y)=(width, height) to (y,x)=(rows,columns) used by numpy
        input_shape = (input_shape[1], input_shape[0], input_shape[2])
        self.frame_skip = frame_skip
        if "doom" in env.lower():
            self.env = WrapperDoom.WrapperDoom(config_file_path=os.path.join(self.root_path, config_file_path),
                                               input_shape=input_shape, frame_skip=self.frame_skip)
        else:
            self.env = WrapperGym.WrapperGym(env, input_shape=input_shape, include_score=include_score,
                                             frame_skip=self.frame_skip)

        # Total number of frames that the simulation will run
        self.num_simul_frames = num_simul_frames
        # Counter to the total number of steps
        self.steps_cont = 0
        # Total number of actions possible inside the environment
        self.actions_num = self.env.numberOfActions()
        # input's shape
        self.input_shape = input_shape
        # Defining the size of the input's third dimension
        self.input_depth = self.input_shape[2]
        # Flag to know if the models has any recurrent layer
        self.is_recurrent = is_recurrent
        # Defining the input of the network
        if self.is_recurrent:
            self.state_input_shape = (history_size, self.input_shape[0], self.input_shape[1], self.input_depth)
        else:
            self.state_input_shape = (self.input_shape[0], self.input_shape[1], (self.input_depth * history_size))
        self.load_weights = load_weights
        self.weights_load_path = weights_load_path
        # =========Learning Parameters===========#
        self.discount_rate = discount_rate
        self.lr = lr
        self.epsilon = epsilon
        self.e_min = e_min
        self.e_exp_decay = e_exp_decay
        self.e_lin_decay = e_lin_decay
        self.decay_mode = decay_mode
        self.target_update = target_update
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.network_model = network_model
        self.normalize_input = normalize_input
        self.Q_value = self.initalize_network(self.network_model, "Q_value")
        self.Q_hat = self.initalize_network(self.network_model, "Q_hat")
        # Initializing the graph an its variables.
        self.initialize_graph()
        # Loading the weights
        if (self.load_weights):
            printd("Loading the Model {}!".format(self.weights_load_path), lvl=2)
            self.Q_value.load_weights(self.weights_load_path)
        # Copying the weights of one NN to another
        self.update_Q_hat()

        # Clipping the error between the interval of 0.0 and 1.0 to compute the Huber Loss
        self.error_clip = 1.0

        # Inicializing the Replay memory
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(num_states_stored=num_states_stored, batch_size=batch_size,
                                          path_save=os.path.join(self.root_path, path_save_episodes),
                                          history_size=history_size, input_shape=input_shape,
                                          is_recurrent=self.is_recurrent)
        self.history_size = history_size
        # Number of random plays to fill the replay memory before the RL-algorithm begins
        self.num_random_play = num_random_play

        # ======Auxiliary variables to plot or control=======#
        self.start_episode = 0
        self.i_episode = 0
        self.loss_value = 0.0
        self.q_rate = 0.0
        self.values_dict = {"Rewards": [], "Loss": [], "Q_value": [], "Num_frames": [],
                            "Time": [], "FPS": [], "Epsilon": []}
        self.image_array = []

        self.steps_save_weights = steps_save_weights
        self.steps_save_plot = steps_save_plot
        self.to_save_episodes = to_save_episodes
        self.steps_save_episodes = steps_save_episodes
        # Checking if the default paths exists.
        if path_save_episodes == "Episodes":
            self.path_save_episodes = os.path.join(self.root_path, path_save_episodes)
            folder_exists(self.path_save_episodes)
        if path_save_plot == "Plot":
            self.path_save_plot = os.path.join(self.root_path, path_save_plot)
            folder_exists(self.path_save_plot)
        if path_save_weights == "Weights":
            self.path_save_weights = os.path.join(self.root_path, path_save_weights)
            folder_exists(self.path_save_weights)

        # Creating a log file
        self.LOG_FILENAME = os.path.join(self.path_save_plot, '{}-Training-{}.txt'.format(self.agent_name,
                                                                                          self.env.getName()))
        # Just opening and cleaning previous files from another simulations
        with open(self.LOG_FILENAME, "w") as text_file:
            pass
        self.summary()

        # Multi-threading variables
        self.st = 0
        self.act = 0
        self.r = 0
        self.st_next = 0
        self.d = 0
        self.queue_ready = False
        self.run_thread = True
        self.thread_sample = threading.Thread(target=self.sample_queue)

    def summary(self):
        """
        Function that display this object information.

        :return:
            nothing
        """
        strr = ""
        strr += "\n============================================================================================"
        strr += "\nINITIALIZING THE DQN ALGORITHM WITH THE FOLLOWING SETTINGS:"
        strr += "\n\tMODE: {}".format(self.mode.upper())
        strr += "\n\tStart Time: {}".format(time.strftime("%d %b %Y %H:%M:%S", time.localtime()))
        strr += "\n\tEnvironment: {}".format(self.env.getName())
        strr += "\n\tNetwork architecture: {}".format(self.network_model)
        strr += "\n\tNormalize input: {}".format(self.normalize_input)
        strr += "\n\tFrame shape: {}".format(self.input_shape)
        strr += "\n\tHistory size: {}".format(self.history_size)
        strr += "\n\tState shape: {}".format(self.state_input_shape)
        if self.is_recurrent:
            strr += "\n\tThe model has a recurrent architecture"
        if self.mode == "train":
            strr += "\n\tTotal number of frames to be be simulated: {} frame(s)".format(self.num_simul_frames)
            strr += "\n\tDiscount rate: {}".format(self.discount_rate)
            strr += "\n\tInitial Epsilon: {}".format(self.epsilon)
            strr += "\n\tFinal Epsilon: {}".format(self.e_min)
            if self.decay_mode.lower() == "linear":
                strr += "\n\tLinear Decay mode is activated!"
                strr += "\n\tThe final Epsilon will be reached in: {} frame(s)".format(self.e_lin_decay)
            else:
                strr += "\n\tExponential Decay mode is activated!"
                strr += "\n\tThe final Epsilon will be reached in approximately: {} episode(s)" \
                    .format(self.e_exp_decay * 5)
            strr += "\n\tLearning rate: {}".format(self.lr)
            strr += "\n\tBatch size: {}".format(self.batch_size)
            strr += "\n\tThe Network will have the {} loss".format(self.loss_type.upper())
            strr += "\n\tThe Network will be trained using {} optimizer".format(self.optimizer.upper())
            strr += "\n\tThe Target Network will be updated every: {} frame(s)".format(self.target_update)
            strr += "\n\tThe Replay Memory will store: {} state(s)".format(self.replay_memory.num_states_stored)
            strr += "\n\tApproximated number of states from random plays before training: {} state(s)" \
                .format(self.num_random_play)
        if self.to_save_episodes:
            strr += "\n\tThe episode will be saved in: {}".format(self.path_save_episodes)
            strr += "\n\tAn episode will be saved each {} episodes".format(self.steps_save_episodes)
        strr += "\n\tThe information will be saved in: {}".format(self.path_save_plot)
        strr += "\n\tThe plot variables will be saved each: {} frame(s)".format(self.steps_save_plot)
        if self.load_weights:
            strr += "\n\tLoad the weights is set to True!"
            strr += "\n\tThe weights will be loaded from : {}".format(self.weights_load_path)
        if self.mode == "train":
            strr += "\n\tThe neural network's weights will be saved in: {}".format(self.path_save_weights)
            strr += "\n\tThe weights will be saved each: {} frame(s)".format(self.steps_save_weights)
        strr += "\n\tMulti gpu mode : {}".format(self.multi_gpu)
        if not self.multi_gpu:
            strr += "\n\tID from the GPU device used : {}".format(self.gpu_device)
        strr += "\n\tMulti threading mode : {}".format(self.multi_threading)
        strr += "\n============================================================================================"
        printd(strr)
        with open(self.LOG_FILENAME, "a+") as text_file:
            print(strr, file=text_file)

    def initalize_network(self, network_model, name):
        """
        Function that creates the neural network and the Tensorflow session.

        This function creates the Neural Network (NN) that will be used by the agent. The architecture of the
        NN is set by the parameter 'network_model" that corresponds to the name of the method, inside the
        Networks.py file, that creates the network.

        :param  network_model : str
                Name of the method, inside the Networks.py file, that creates the network.

        :param  name : str
                The NN's name

        :return nothing
        
        """
        model = getattr(Networks, network_model)(self.state_input_shape, self.actions_num, name, self.normalize_input)
        self.sess = tf.Session()
        K.set_session(self.sess)
        if (utils.DEBUG and utils.DEBUG_lvl >= 1):
            model.summary()
        return model

    def initialize_graph(self):
        """
        Function that initializes the tensorflow graph that computes the NN training.

        This function creates the tensorflow graph that will compute the NN error and training. The network
        can be trained by two different losses, huber loss (default) and mse (Mean Squared Error), the type of
        loss used is defined on the initialization of this class by the parameter "loss_type". This function
        allows you to choose between two optimizers: RMSProp (with the settings given by the DQN Paper) and
        ADAM (with tensorflow defaults arguments), this choice is also made on the initialization by the 
        parameter "optimizer".
        OBS: The tensorflow graph is fed by dictionary in the function train_dqn().

        :param  nothing

        :return nothing

        """
        # Defining the tensors variables (placeholders)
        self.state = tf.placeholder(tf.uint8, [None] + list(self.state_input_shape))
        self.action = tf.placeholder(tf.int32, [None])
        self.reward = tf.placeholder(tf.float32, [None])
        self.state_next = tf.placeholder(tf.uint8, [None] + list(self.state_input_shape))
        self.done = tf.placeholder(tf.float32, [None])
        # Pre-processing the state
        state_float = tf.cast(self.state, tf.float32)
        state_next_float = tf.cast(self.state_next, tf.float32)
        # Operations
        act_one_hot = tf.one_hot(self.action, self.actions_num, on_value=1.0, off_value=0.0)
        self.mask_one = tf.ones_like(act_one_hot, tf.float32)
        current_q = tf.reduce_sum(self.Q_value([state_float]) * act_one_hot, axis=1)
        prediction = self.Q_hat([state_next_float]) * self.mask_one
        target_q = tf.reduce_max(prediction, axis=1)
        # Computing the NN ERROR as described in the DQN paper.
        target_val = tf.stop_gradient(self.reward + (self.discount_rate * target_q) * (1 - self.done))
        if "huber" in self.loss_type:
            # Computing the Huber Loss
            self.loss_train = tf.losses.huber_loss(current_q, target_val)
        elif "mse" in self.loss_type:
            # Computing the MSE loss
            self.loss_train = tf.losses.mean_squared_error(target_val, current_q)
        if "rms" in self.optimizer.lower():
            # Using RMSprop with DQN paper's parameters
            self.train_op = tf.train.RMSPropOptimizer(
                self.lr, decay=0.95, momentum=0.0, epsilon=0.01).minimize(self.loss_train)
        elif "adam" in self.optimizer.lower():
            # Using the Adam  optimizer with default parameters
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_train)
        # Initializing the variables
        self.sess.run(tf.global_variables_initializer())

    def update_Q_hat(self):
        """
        Function that updates the values of Q_hat with the values of Q_value at each N (:param target_update)
        steps.
        
        :param  nothing
        :return nothing
        
        """
        self.Q_hat.set_weights(self.Q_value.get_weights())

    def e_greddy_action(self, state, random_fill=False):
        """
        Function that selects an action with base on the e-greedy police.

        :param  state : input volume (np.array) of shape state_input_shape (dtype=np.int8)
                    A volume compound of a set of states (images) of depth "history_size".

        :return nothing
        """

        # Gets a random action if the variable self.epsilon is less than a random variable
        # (distributed between 0 and 1)
        if not self.mode.lower() == "test" and (random_fill or np.random.random() < self.epsilon):
            action = np.random.choice(np.arange(self.actions_num))
        # Otherwise the algorithm computes the Q value of each action possible for that state and
        # pick the one with maximal value.
        else:
            # Reshaping the state to add one axis to send it as input to the Neural Network
            state = state.reshape((1,) + state.shape)
            prediction = self.Q_value.predict_on_batch([state])
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
        if self.decay_mode.lower() == "linear":
            # straight line equation wrapper by max operation -> max(min_value,(-mx + b))
            self.epsilon = np.amax((self.e_min, -((1.0 - self.e_min) * self.steps_cont) / self.e_lin_decay + 1.0))
        else:
            # exponential's function Const(e^-t) wrapped by a min function
            self.epsilon = np.amin((1, (self.e_min + (1.0 - self.e_min) * np.exp(-(self.i_episode - 1) /
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
        # Uniformly sampling from the replay memory
        if self.multi_threading:
            lock.acquire(blocking=True)
            st = self.st
            act = self.act
            r = self.r
            st_next = self.st_next
            d = self.d
            self.queue_ready = False
            lock.release()
        else:
            st, act, r, st_next, d, idx = self.replay_memory.sample()
        self.loss += self.sess.run([self.train_op, self.loss_train],
                                   feed_dict={self.state: st, self.action: act, self.reward: r,
                                              self.state_next: st_next, self.done: d})[1]

    def save_gif(self, saved_gif, name="", path_save_gif=""):
        """
        Function that saves an episode (can be a state) as .gif file.

        :param  saved_gif : np.array (dtype=np.uint8).
                    Sequence of frames concatenated in a np.array (dtype=np.uint8).

        :return nothing

        """

        if name == "":
            name = "{}-{}-{}-Episode-{}.gif".format(self.agent_name, self.mode, self.env.getName(),
                                                    self.i_episode)
        if path_save_gif == "":
            path_save_gif = os.path.join(self.path_save_episodes, name)
        else:
            path_save_gif = os.path.join(path_save_gif, name)
        if not self.is_recurrent:
            n_frames = saved_gif.shape[2] / (self.input_depth)
            saved_gif = np.split(saved_gif, n_frames, axis=2)
        imageio.mimwrite(path_save_gif, saved_gif, fps=60)

    def save_weights(self):
        """
        Function that saves the weights in .h5 file. The weights are saved each N steps
        (defined by steps_save_weights on initialization).
        
        :param  nothing

        :return nothing

        """
        self.Q_value.save_weights(os.path.join(self.path_save_weights,
                                               "{}-weights-{}-{}.h5".format(self.agent_name, self.env.getName(),
                                                                            self.steps_cont)))

    def save_plot(self):
        """
        Function that saves the variables to plot in a .csv file. The variables
        are saved each N steps (defined by steps_save_plot on initialization).

        :param  nothing

        :return nothing

        """
        df = pd.DataFrame.from_dict(self.values_dict)
        df.to_csv(os.path.join(self.path_save_plot, '{}-{}-{}.csv'.format(self.agent_name, self.mode,
                                                                          self.env.getName())), index=False)

    def refresh_history(self, history, state_next):
        """
        Function that updates the history (a set of "n" frames that is used as a state of the replay memory)
        taking out the first frame, moving the rest and adding the new frame to end of the history.

        :param history : input volume of shape state_input_shape
                The history that will be refreshed (basically a set of n frames concatenated
                [np.array dtype=np.int8]) as a state on the replay memory.

        :param state_next : Image (np.array of dtype=np.uint8 of input_shape)
                Frame (np.array dtype=np.int8) of the environment's current state after a action was taken.

        :return nothing
        """
        if self.is_recurrent:
            history[:-1] = history[1:]
            history[-1] = state_next
        else:
            history[:, :, :-self.input_depth] = history[:, :, self.input_depth:]
            history[:, :, -self.input_depth:] = state_next
        return history

    def reshape_state(self, state):
        ax = 2
        if self.is_recurrent:
            state_concat = [np.expand_dims(state, axis=0) for _ in range(self.history_size)]
            ax = 0
        else:
            state_concat = [state for _ in range(self.history_size)]
            # Transforming the receive state (image frame) in a volume of n frames (history)
        state_aux = np.concatenate(state_concat, axis=ax)
        return state_aux

    def sample_queue(self):
        """
        Function that is executed in a separate thread that samples from the replay memory and let the samples
        ready for the train method (thus, cutting the execution time).

        :param  nothing

        :return nothing

        """
        while self.run_thread:
            lock.acquire(blocking=True)
            if not self.queue_ready and self.replay_memory.size() >= self.batch_size:
                self.st, self.act, self.r, self.st_next, self.d, _ = self.replay_memory.sample()
                self.queue_ready = True
            lock.release()
            # Sleep time to give the other thread time to get access to the lock object
            time.sleep(1e-8)

    def summary_run(self, t, reward_total_episode, fps, time_it, mode="random_fill"):
        """
        Function that computes the plot variables and displays the information of the execution mode
        to the user.

        :param  t : int
                Number of time steps of an ended episode.

        :param  reward_total_episode : float
                Total reward of the episode.

        :param  fps : float
                Number of frames per second (fps) processed from the episode.

        :param  time_it : float (time)
                Variable that stores the execution time.

        :param  mode : str (Default "random_fill")
                Execution mode.

        :return nothing

        """
        avg_loss = 0
        avg_q_rate = 0
        if mode == "train":
            avg_loss = self.loss / (t + 1)
            avg_q_rate = self.q_rate / (t + 1)
            self.values_dict["Rewards"].append(reward_total_episode)
            self.values_dict["Loss"].append(avg_loss)
            self.values_dict["Q_value"].append(avg_q_rate)
            self.values_dict["Num_frames"].append(self.steps_cont)
            self.values_dict["Time"].append(time.time() - time_it)
            self.values_dict["FPS"].append(fps)
            self.values_dict["Epsilon"].append(self.epsilon)

        strr = ""
        strr += "Episode {:d}:".format(self.i_episode)
        strr += "\n\t\t\tTotal Frames: {:d}/{:d},".format(self.steps_cont, self.num_simul_frames)
        strr += "\n\t\t\tFrames in this episode: {:d},".format(t)
        strr += "\n\t\t\tTotal reward: {:.2f},".format(reward_total_episode)
        if self.mode.lower() == "train":
            strr += "\n\t\t\tEpsilon: {:.4f},".format(self.epsilon)
            strr += "\n\t\t\tReplay memory size: {:d}/{:d},".format(self.replay_memory.size(),
                                                                    self.replay_memory.num_states_stored)
            strr += "\n\t\t\tLoss: {:.4f},".format(avg_loss)
            strr += "\n\t\t\tMean Q value: {:.4f},".format(avg_q_rate)
        strr += "\n\t\t\tFPS: {:.2f}, ".format(fps)
        strr += "\n\t\t\tTime of this episode: {:.3f} (s)".format(time.time() - time_it)
        with open(self.LOG_FILENAME, "a+") as text_file:
            print(strr, file=text_file)
        printd(strr)

    def run_random_fill(self):
        """
        Function that fills the replay memory with states that come from actions chosen randomly.

        :param  nothing

        :return nothing

        """
        self.steps_cont = 0
        time_it = time.time()
        self.i_episode = 0
        while self.replay_memory.size() < self.num_random_play:
            self.i_episode += 1
            state = self.env.reset()
            state = self.reshape_state(state)
            # ======Initializing variables====#
            done = False
            t = 0
            reward_total_episode = 0
            while not done:
                t += 1
                # accomulate the total number of frames
                self.steps_cont += 1
                action = self.e_greddy_action(state, random_fill=True)
                state_next, reward, done, _ = self.env.step(action)
                # Updating the input volume to put the current next_state
                state_next = self.refresh_history(np.copy(state), state_next)
                self.replay_memory.append(state, action, reward, state_next, done)
                state = np.copy(state_next)
                reward_total_episode += reward

            # Saving the variables to plot and the episode
            fps = t / (time.time() - time_it)
            self.summary_run(t=t, reward_total_episode=reward_total_episode, fps=fps, time_it=time_it)
            time_it = time.time()

        self.env.close()

    def run_train(self, to_render=False):
        """
        Function that trains the RL-DQN algorithm as demonstrated in the DQN paper.

        :param  to_render : bool (default False)
                Variable that decides if it's to render on the screen the current episode.
                OBS: If this variable is true the fps will decrease harshly since it needs to
                show the game in reasonable speed.

        :return nothing

        """
        self.env.render(to_render)
        self.steps_cont = 0
        time_it = time.time()
        self.i_episode = 0
        saved_episode = 0
        if self.multi_threading:
            self.thread_sample.start()
        while self.steps_cont < self.num_simul_frames:
            self.i_episode += 1
            state = self.env.reset()
            # Starting to save the episode (if it's to save)
            if self.to_save_episodes and self.i_episode % self.steps_save_episodes == 0:
                if self.is_recurrent:
                    saved_episode = np.expand_dims(state, axis=0)
                else:
                    saved_episode = state
            state = self.reshape_state(state)
            # ======Initializing variables====#
            done = False
            t = 0
            self.loss = 0
            self.q_rate = 0
            reward_total_episode = 0
            while not done:
                t += 1
                # accumulate the total number of frames
                self.steps_cont += 1
                action = self.e_greddy_action(state, random_fill=False)
                state_next, reward, done, _ = self.env.step(action)
                if self.to_save_episodes and self.i_episode % self.steps_save_episodes == 0:
                    if self.is_recurrent:
                        saved_episode = np.concatenate((saved_episode, np.expand_dims(state_next, axis=0)),
                                                       axis=0)
                    else:
                        saved_episode = np.concatenate((saved_episode, state_next), axis=2)
                # Updating the input volume to put the current next_state
                state_next = self.refresh_history(np.copy(state), state_next)
                if self.multi_threading:
                    lock.acquire(blocking=True)
                    self.replay_memory.append(state, action, reward, state_next, done)
                    lock.release()
                else:
                    self.replay_memory.append(state, action, reward, state_next, done)
                state = np.copy(state_next)
                reward_total_episode += reward
                # 2* to make sure that the sampling thread executes first case there's no random_fill
                if self.replay_memory.size() > 2 * (self.batch_size):
                    self.train_dqn()
                    if self.epsilon > self.e_min:
                        self.decay_epsilon()
                    if (self.steps_cont % self.target_update == 0):
                        printd("Q_hat was renewed!", lvl=2)
                        self.update_Q_hat()
                    if (self.steps_cont % self.steps_save_weights == 0):
                        self.save_weights()
                    if (self.steps_cont % self.steps_save_plot == 0):
                        self.save_plot()

            # Saving the episode
            if self.to_save_episodes and self.i_episode % self.steps_save_episodes == 0:
                self.save_gif(saved_episode)
            fps = t / (time.time() - time_it)
            self.summary_run(mode="train", t=t, reward_total_episode=reward_total_episode, fps=fps,
                             time_it=time_it)
            time_it = time.time()

        self.run_thread = False
        self.env.close()

    def run_test(self, to_render=True, to_save_states=False, path_save_states="States"):
        """
        Function that runs a test with the weights loaded from a previous simulation.

        :param  to_render : bool (default True)
                Variable that decides if it's to render on the screen the current episode.
                OBS: If this variable is true the fps will decrease harshly since it needs to
                show the game in reasonable speed.

        :return nothing

        """
        # Checking if the default path to save the states exists if not creates it.
        if to_save_states and path_save_states == "States":
            self.path_save_plot = os.path.join(self.root_path, path_save_states)
            folder_exists(self.path_save_plot)
        self.env.render(to_render)
        self.steps_cont = 0
        time_it = time.time()
        self.i_episode = 0
        while self.steps_cont < self.num_simul_frames:
            # ======Initializing variables====#
            done = False
            t = 0
            self.loss = 0
            self.q_rate = 0
            reward_total_episode = 0
            self.i_episode += 1
            state = self.env.reset()
            state = self.reshape_state(state)
            if to_save_states:
                name_str = "{}-{}-{}-Episode-{}-State-{}.gif".format(self.agent_name, self.mode,
                                                                     self.env.getName(), self.i_episode, t)
                self.save_gif(state, name=name_str, path_save_gif=path_save_states)

            while not done:
                t += 1
                # accumulate the total number of frames
                self.steps_cont += 1
                action = self.e_greddy_action(np.copy(state), random_fill=False)
                state_next, reward, done, _ = self.env.step(action)
                # Updating the input volume to put the current next_state
                state_next = self.refresh_history(np.copy(state), state_next)
                state = np.copy(state_next)
                if to_save_states:
                    name_str = "{}-{}-{}-Episode-{}-State-{}.gif".format(self.agent_name, self.mode,
                                                                         self.env.getName(), self.i_episode, t)
                    self.save_gif(state, name=name_str, path_save_gif=path_save_states)
                reward_total_episode += reward
                # Sleep time to make the render in a reasonable speed(not to fast).
                time.sleep(1 / 100)
            fps = t / (time.time() - time_it)
            self.summary_run(mode="test", t=t, reward_total_episode=reward_total_episode, fps=fps,
                             time_it=time_it)
            time_it = time.time()

        self.env.close()


def agent_arg_parser(parser):
    str = None
    # If no argument was sent, use the <this_file_name>.cfg file
    if len(sys.argv) == 1:
        # Replacing the .py from this file for .cfg, thus appointing to the file with the configurations
        file_cfg = sys.argv[0][:-3] + ".cfg"
        # If the file exists, read it
        if (os.path.exists(file_cfg)):
            str = read_cfg(file_cfg)
    parser.add_argument("--agent_mode", choices=["train", "test"], default="train",
        help="Mode to execute the algorithm. Type:str. Default: train")
    parser.add_argument("--agent_name", default="DQN",
        help="Agent's name, it will be passed to the saved files (weights,episodes,plot). Type:str. "
                             "Default: DQN")
    parser.add_argument("--env", default='PongNoFrameskip-v4',
        help=" The name of the environment where the agent will interact. Type: str."
                             " Default:PongNoFrameskip-v4")
    parser.add_argument("--include_score", default=False, type=str2bool,
        help="If its to include in the state image the score from the environment (atari game)."
                             " Type: bool. Default: False. [GYM ATARI EXCLUSIVE]")
    parser.add_argument("--config_file_path", default="DoomScenarios/labyrinth.cfg",
        help="Path to .cfg file that contains the configuration to the Doom's environment. Type: str. "
                             "Default:../DoomScenarios/labyrinth.cfg")
    parser.add_argument("--network_model", default="DQN",
        help="Neural Network's architecture to be used. The name should match one of the methods inside the "
        "Networks.py. You can create you own model inside the Networks.py and send the name of the method "
        "to this argument. Type; str. Default:DQN")
    parser.add_argument("--normalize_input", default=True, type=str2bool,
        help="Flag that controls if it's to save episodes on the disk. Type:bool. Default:True")
    parser.add_argument("--frame_skip", default=4, type=int,
        help=" Total number of frames that will be skipped between states.. Type:int. Default:4")
    parser.add_argument("--num_simul_frames", default=10000000, type=int,
        help="Total number of frames that the agent will be trained. Type:int. Default:10000000")
    parser.add_argument("--discount_rate", default=0.99, type=float,
        help="Discount rate gamma. Type: float. Default:0.99")
    parser.add_argument("--lr", default=0.00025, type=float,
        help="Neural Network learning rate. Type: float. Default:0.00025")
    parser.add_argument("--epsilon", default=1.0, type=float,
        help="Probability's initial value of the agent to choose random actions (exploration) using the "
        "policy e-greedy. Type:float. Default:1.0")
    parser.add_argument("--e_min", default=0.1, type=float,
        help="Probability's final value of the agent to choose random actions (exploration) using the "
        "policy e-greedy. Type:float. Default:0.1")
    parser.add_argument("--decay_mode", default="linear", choices=["linear", "exponential"],
        help="Type of epsilon's decay mode. There are two possible types: \"linear\" and \"exponential\". "
        "Type: str. Default: linear")
    parser.add_argument("--e_lin_decay", default=1000000, type=int,
        help="Number of frames for epsilon reach its final value linearly (e_min). Type: int. Default:1000000")
    parser.add_argument("--e_exp_decay", default=300, type=int,
        help="Exponential decay rate in EPISODES (The decay is slowly with bigger values since the decay"
        "equation is exp^[-1/e_exp_decay]). Type:int. Default:300")
    parser.add_argument("--target_update", default=10000, type=int,
        help="Number of frames that the parameters of Q_target will be updated with the parameters of Q"
        "[See the DQN paper for more details]. Type:int. Default:10000")
    parser.add_argument("--num_states_stored", default=1000000, type=int,
        help="Number of states stored in the replay memory. Type:int. Default:1000000")
    parser.add_argument("--batch_size", default=32, type=int,
        help="The batch's size to train the NN. Type: int. Default:32")
    parser.add_argument("--input_shape", default="84 84",
        help="Input frame's shape (WxHxColor_channel[if any]) that will be sent to the Neural Network. If "
        "just WxH are entered, the color_channel will be 1 (gray_scale)"
        "Type:str (with each argument separated by space or comma, and the whole sentence between quotation "
        "marks). Default:\"84 84\"")
    parser.add_argument("--history_size", default=4, type=int,
        help="Number of sequential frames that will be stacked together to form the input volume to the NN. "
        "Type:int. Default:4")
    parser.add_argument("--num_random_play", default=50000, type=int,
        help="Number of states generated by actions chosen randomly that will be stored in the replay memory"
        "before the agent's training begins. Type:int. Default:50000")
    parser.add_argument("--load_weights", default=False, type=str2bool,
        help="Variable that controls if it's to load the weights from a external .h5 file generated by "
        "another simulation. Type:bool. Default (Train): False. Default(Test): False")
    parser.add_argument("--steps_save_weights", default=50000, type=int,
        help="Desired number of frames to save the weights. Type:int. Default: 50000")
    parser.add_argument("--steps_save_plot", default=10000, type=int,
        help="Desired number of frames to save the plot variables. Type:int. Default:10000")
    parser.add_argument("--to_save_episodes", default=False, type=str2bool,
        help="Flag that controls if it's to save episodes on the disk. Type:bool. Default:False")
    parser.add_argument("--steps_save_episodes", default=50, type=int,
        help="Number of episodes that an episode will be saved on the disk as .gif. Type: int. Default:50")
    parser.add_argument("--path_save_episodes", default="Episodes",
        help="Path to the folder where will be saved the episode as .gif file. Type: str. Default:\"Episodes\"")
    parser.add_argument("--weights_load_path", default="",
        help="Path to .h5 file that contains the pre-treined weights. Default: None. REQUIRED IN TEST MODE")
    parser.add_argument("--loss_type", default="huber", choices=["huber", "mse"],
        help="Name of the type of loss function that will be used to train the DQN Network. There are two "
        "possible types: \"huber\" and \"MSE\". Type: str. Default: \"huber\"")
    parser.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "adam"],
        help="Name of the type of optimizer that will be used to train the DQN Network. There are two possible "
        "types: \"rmsprop\" and \"adam\". The first one uses the setting described on the DQN paper, "
        "the second uses the tensorflow/keras default parameters. Type:str. Default:\"rmsprop\"")
    parser.add_argument("--path_save_plot", default="Plot",
        help="Folder's path where will be saved the .csv file with the algorithm's information. Type:str."
        "Default:\"<this_folder>\\Plot\"")
    parser.add_argument("--path_save_weights", default="Weights",
        help="Folder's path where will be saved the .h5 file with the Neural Network Weights. Type:str. "
        "Default:\"<this_folder>\\Weights\"")
    parser.add_argument("--silent_mode", default=False, type=str2bool,
        help="If it's active no message will be displayed on the prompt. Type:bool. Default:False.")
    parser.add_argument("--multi_gpu", default=False, type=str2bool,
        help="If false, you can select what gpu to use (if there is more than one). Type:bool. Default:False")
    parser.add_argument("--gpu_device", default=0, type=int,
        help="The ID of the gpu device that will be used in the case of the multi_gpu variable "
        "is False and there is multiple GPUs. Type:int. Default:0")
    parser.add_argument("--multi_threading", default=False, type=str2bool,
        help="If this mode is active the sampling part of the algorithm will be done in parallel with the main"
        " RL-algorithm. Type:bool. Default:False")
    parser.add_argument("--to_render", default=False, type=str2bool,
        help="If this mode is active the environment will be rendered. Type:bool. Default:False")
    parser.add_argument("--to_save_states", default=False, type=str2bool,
        help="Controls if it to save the states in TEST MODE. Type:bool. Default:False")
    parser.add_argument("--path_save_states", default="States",
        help="Folder's path where will be saved the state as .gif images. Type:str. Default : States. "
        "Default:\"<this_folder>\\States\" TEST MODE ONLY")
    parser.add_argument("--is_recurrent", default=False, type=str2bool,
        help="If your model has any recurrent layer set this flag to True. Type:bool. Default:False")
    parser.add_argument("--random_seed", default=-1, type=int,
        help="Seed to the random methods used by this agent. If the value is -1. No seed is set at all."
        " Type:int. Default:-1")
    if str is not None:
        args = parser.parse_args(str)
    else:
        args = parser.parse_args()
    # Parsing the input_shape argument to int
    input_shape_aux = tuple([int(item) for item in re.findall(r"\d+", args.input_shape)])
    # Parsing the arguments as a dict
    kwargs = {"agent_name": args.agent_name,
              "mode": args.agent_mode,
              "env": args.env,
              "config_file_path": args.config_file_path,
              "network_model": args.network_model,
              "normalize_input": args.normalize_input,
              "frame_skip": args.frame_skip,
              "num_simul_frames": args.num_simul_frames,
              "discount_rate": args.discount_rate,
              "lr": args.lr,
              "epsilon": args.epsilon,
              "e_min": args.e_min,
              "decay_mode": args.decay_mode,
              "e_lin_decay": args.e_lin_decay,
              "e_exp_decay": args.e_exp_decay,
              "target_update": args.target_update,
              "num_states_stored": args.num_states_stored,
              "batch_size": args.batch_size,
              "input_shape": input_shape_aux,
              "history_size": args.history_size,
              "num_random_play": args.num_random_play,
              "load_weights": args.load_weights,
              "steps_save_weights": args.steps_save_weights,
              "steps_save_plot": args.steps_save_plot,
              "to_save_episodes": args.to_save_episodes,
              "steps_save_episodes": args.steps_save_episodes,
              "path_save_episodes": args.path_save_episodes,
              "weights_load_path": args.weights_load_path,
              "loss_type": args.loss_type,
              "optimizer": args.optimizer,
              "path_save_plot": args.path_save_plot,
              "path_save_weights": args.path_save_weights,
              "silent_mode": args.silent_mode,
              "multi_gpu": args.multi_gpu,
              "gpu_device": args.gpu_device,
              "multi_threading": args.multi_threading,
              "is_recurrent": args.is_recurrent}
    if args.agent_mode.lower() == "test":
        assert args.load_weights == True, "In test mode you have to specify the path to load " \
                                          "the pre-trained weights"
    return args, kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a RL-agent in an environment.")
    args, kwargs = agent_arg_parser(parser)
    dqn = Agent(**kwargs)
    if args.random_seed != -1:
        # Setting all random seeds to a value known
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        set_random_seed(args.random_seed)
        dqn.env.set_seed(args.random_seed)
    if args.agent_mode.lower() == "train":
        if args.num_random_play > 0:
            printd("EXECUTING RANDOM PLAYS TO FILL THE REPLAY MEMORY")
            dqn.run_random_fill()
        printd("EXECUTING AND TRAINING DQN ALGORITHM")
        dqn.run_train(to_render=args.to_render)
    elif not args.weights_load_path == "":
        dqn.run_test(to_render=args.to_render,
                     to_save_states=args.to_save_states,
                     path_save_states=args.path_save_states)
    else:
        raise Exception("The path to load the weights was not valid!")
