################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################

from Environments import Base
import cv2
import vizdoom as vzd
import re
import numpy as np

class WrapperDoom(Base):

    def __init__(self,config_file_path,input_shape=(84,84)):
        self.env = vzd.DoomGame()
        self.config_file_path = config_file_path
        self.env.load_config(self.config_file_path)
        self.input_shape = input_shape
        self.game_init = False
        self.actions = self.one_hot_actions()

    def getName(self):
        aux=re.findall("\w*.cfg",self.config_file_path)[0][:-4]
        return "Doom-{}".format(aux)

    def reset(self):
        if not self.game_init:
            self.env.init()
            self.game_init = True
        self.env.new_episode()
        return self.pre_process(self.env.get_state().screen_buffer)

    def render(self, is_to_render):
        self.env.set_window_visible(is_to_render)

    def step(self, action):
        reward=self.env.make_action(self.actions[action], tics=4)
        state_next = self.pre_process(self.env.get_state().screen_buffer)
        done = self.env.is_episode_finished()
        return (state_next,reward,done,"")

    def numberOfActions(self):
        return len(self.env.get_available_buttons())

    def set_seed(self, seed):
        self.env.set_seed(seed)

    def close(self):
        self.env.close()
        self.game_init = False

    def pre_process(self,image):
        img=cv2.resize(image,dsize=self.input_shape)
        return img

    def one_hot_actions(self):
        aux_one_hot = []
        for i in range(self.numberOfActions()):
            aux = np.zeros((self.numberOfActions()), dtype=np.uint8)
            aux[i] = 1
            aux_one_hot.append(aux.tolist())
        return aux_one_hot