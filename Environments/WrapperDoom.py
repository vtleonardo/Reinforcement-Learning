################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################

from Environments.Base import Base
import cv2
import vizdoom as vzd
import re
import numpy as np

class WrapperDoom(Base):
    """
    Wrapper for the Doom environment to match all methods required by the RL-algorithm.

    This class inherit the Base class, therefore it have the methods:
        getName: returns a string with the environment's name
        render:  make the environment's rendering
        reset: resets the environment and return the state
        step: does an action on the environment and returns its consequences
                    (the reward, the next state, if the episode has done, and other infos).
        numberOfActions: returns the number of actions possible on this environment.
        set_seed : set the random seed of the environment
        close : close the environment
    """

    def __init__(self, config_file_path, input_shape=(84,84)):
        """
        Creates the object

        :param config_file_path: str (path)
            Path to .cfg file that contains the configuration to the Doom's environment.
        :param input_shape tuple (int,int)
            Tuple that contains the size (WxH) of the image that will be fed to the Neural Network.
        """
        self.env = vzd.DoomGame()
        self.config_file_path = config_file_path
        self.env.load_config(self.config_file_path)
        self.input_shape = input_shape
        #If the game has began.
        self.game_init = False
        # Vizdoom only accepts actions format of a vector of binaries (ie.010) since it allows multiple inputs
        # at the same time, thus we need to change from int to binary format
        self.actions = self.one_hot_actions()
        self.terminal_state_next = np.expand_dims(np.zeros(input_shape,dtype=np.uint8),axis=2)

    def getName(self):
        """
        Function that gets the environment's name

        :return: str
            String with the environment's name in the format "Doom-Map loaded"
        """
        aux=re.findall("\w*.cfg",self.config_file_path)[0][:-4]
        return "Doom-{}".format(aux)

    def render(self, is_to_render=False):
        """
        Controls if it's to display the environment on the screen to the user.

        :param is_to_render : bool (Default : False)
                Flag that controls if it's to display the environment on the screen to the user.
        :return:
            nothing
        """
        self.env.set_window_visible(is_to_render)

    def reset(self):
        """
        Resets the environment ant its variables and send back the initial state.

        :return: numpy.array [dtype=uint8] with shape (input_shape from you RL_algorithm)
            initial state (frame image [np.array of dtype=uint8]) of the environment
        """
        # Vizdoom only allows to change games setting before we start the game, thus we need to close a section
        # and start another to change the render setting.
        if not self.game_init:
            self.env.init()
            self.game_init = True
        self.env.new_episode()
        return self.pre_process(self.env.get_state().screen_buffer)

    def step(self, action):
        """
        Controls if it's to display the environment on the screen to the user.

        :param is_to_render : bool (Default : False)
                Flag that controls if it's to display the environment on the screen to the user.
        :return:
            nothing
        """
        # Making an action and than skipping 4 frames
        reward=self.env.make_action(self.actions[action],4)
        state=self.env.get_state()
        done = self.env.is_episode_finished()
        if not done:
            state_next = self.pre_process(state.screen_buffer)
        else:
            state_next = self.terminal_state_next
        return (state_next,reward,done,"")

    def numberOfActions(self):
        """
         Function that returns the number os possible actions for this environment.

         :return: int
             The number of possible actions for this environment
         """
        return len(self.env.get_available_buttons())

    def set_seed(self, seed):
        """
        Function that sets the random seed of the environment.

        :param
        seed : int
            Seed to set the random function of this environment

        :return:
            nothing
        """
        self.env.set_seed(seed)

    def close(self):
        """
        Function that closes the current environment.

        :param nothing

        :return: nothing
        """
        self.env.close()
        self.game_init = False

    def pre_process(self,image):
        """
        Resize the image tha comes from the Doom environment to the size required by the RL-algorithm.

        :param image: np.array (dtype=np.uint8)
                Image to be resized

        :return: img: np.array (dtype=np.uint8)
                The image with the new size.
        """
        img=np.expand_dims(cv2.resize(image,dsize=self.input_shape),axis=2)
        return img

    def one_hot_actions(self):
        """
        Vizdoom only accepts actions in a format of a vector of binaries (ie.010) since it allows multiple
        inputs at the same time, thus we need to change from int(format used by the RL-algorithm) to binary
        format.

        :return: vector with actions in the one_hot encode format.
        """
        aux_one_hot = []
        for i in range(self.numberOfActions()):
            aux = np.zeros((self.numberOfActions()), dtype=np.uint8)
            aux[i] = 1
            aux_one_hot.append(aux.tolist())
        return aux_one_hot