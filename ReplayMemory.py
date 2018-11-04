################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################

import numpy as np
from collections import deque
import imageio
from utils import folder_exists
import os

class ReplayMemory():
    """
    Class that stores the experiences lived by the agent.

    In this class each parameter that compose an experience will be stored in a different deque.
    This class only stores a given frame once and after that links its position to the states that compose
    an experience hence saving memory ram.

    """
    def __init__(self,num_states_stored=1000000, batch_size=32, path_save="", history_size=4,
                                                                    input_shape=(84,84,1)):
        """
        Initializes the replay memory, each item in an experience will be stored in a different deque.

        :param num_states_stored : int (Default : 1 000 000)
                    Number of experiences to be stored in the memory.
        :param batch_size : int (Default : 32)
                    Number of samples that will be sampled from the memory and sent back to agent.
        :param path_save : str (Default : "")
                    Path to the folder where the states will be saved.

        """
        self.path_save=path_save
        self.batch_size = batch_size
        self.num_states_stored = num_states_stored
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self._first_state = True
        self.history_size = history_size
        # Variable that stack the frames
        self.stacked_frames = deque(maxlen=history_size*self.input_depth)
        #Deque that stores the frames
        self.frames = deque(maxlen=self.num_states_stored*self.input_depth)
        # DEQUE with fixed length (num_states_stored) for each item inside an experience
        self.replay_memory_state = deque(maxlen=self.num_states_stored)
        self.replay_memory_action = deque(maxlen=self.num_states_stored)
        self.replay_memory_reward = deque(maxlen=self.num_states_stored)
        self.replay_memory_state_next = deque(maxlen=self.num_states_stored)
        self.replay_memory_done = deque(maxlen=self.num_states_stored)
        # DEQUE of length (batch size) to return the sampled items
        self.state_to_return = deque(maxlen=self.batch_size)
        self.action_to_return = deque(maxlen=self.batch_size)
        self.reward_to_return = deque(maxlen=self.batch_size)
        self.state_next_to_return = deque(maxlen=self.batch_size)
        self.done_to_return = deque(maxlen=self.batch_size)

    def append(self,state,action,reward,state_next,done):
        """
        This method's responsible for appending the items that compose an experience.

        :param state: volume of dtype_np.int8 and shape input shape.
                The environment's state before the agent has took an action.

        :param action: int
                The numerical (not one hot encoded) value of an action tha was executed by the agent.

        :param reward: int/float
                The reward given by the environment as result from the agent taking an action.

        :param state_next: volume of dtype_np.int8 and shape input shape.
                The environment's state before the agent has took an action.

        :param done: bool
                A flag that signalizes if a given state state_next is terminal.

        :return: nothing

        """
        if self._first_state:
            self._first_state = False
            for j in range(self.input_depth):
                self.frames.append(state[:, :, -self.input_depth+j].copy())
            # For each initial state we need to stack the first frame.
            for i in range(self.history_size):
                for j in range(self.input_depth):
                    self.stacked_frames.append(self.frames[-self.input_depth+j])
            self.replay_memory_state.append(self.stacked_frames.copy())
        else:
            self.replay_memory_state.append(self.replay_memory_state_next[-1].copy())

        for j in range(self.input_depth):
            self.frames.append(state_next[:, :, -self.input_depth+j].copy())
            self.stacked_frames.append(self.frames[-1])
        self.replay_memory_state_next.append(self.stacked_frames.copy())
        self.replay_memory_action.append(action)
        self.replay_memory_reward.append(reward)
        self.replay_memory_done.append(done*1)
        # If this is a terminal state, the next experience will be a initial state (first episode)
        if done:
            self._first_state=True

    def sample(self):
        """
        Method that samples uniformly N (from size batch_size) elements from the replay memory.

        :return: tuple compose of:
            state : N elements of in a form of np.array where each element has a type of np.uint8

            action : N elements of in a form of np.array where each element has a type of np.int32

            reward : N elements of in a form of np.array where each element has a type of np.float32

            state_next : N elements of in a form of np.array where each element has a type of np.uint8

            done : N elements of in a form of np.array where each element has a type of np.float32
                    (flag(in float type) if this experience is a terminal one)
        """
        # Sampling N indexes of elements uniformly
        idx = np.random.choice(len(self.replay_memory_done), self.batch_size)
        for i in idx:
            self.state_to_return.append(self.get_state(i))
            self.action_to_return.append(self.replay_memory_action[i])
            self.reward_to_return.append(self.replay_memory_reward[i])
            self.state_next_to_return.append(self.get_state_next(i))
            self.done_to_return.append(self.replay_memory_done[i])

        return (np.array(self.state_to_return,dtype=np.uint8),
                np.array(self.action_to_return,dtype=np.int32),
                np.array(self.reward_to_return,dtype=np.float32),
                np.array(self.state_next_to_return,dtype=np.uint8),
                np.array(self.done_to_return,dtype=np.float32),
                idx)


    def size(self):
        """
        Method that gets the current number of elements inside the replay memory.

        :return: size of the replay memory (int)
        """
        return len(self.replay_memory_reward)

    def get_state(self,i):
        """
        Gets a state  from a experience located on index i from inside the replay memory.

        :param  i :  int
                Index of the item (int)

        :return: state (np.array dtype=uint8 of shape input_shape [DQN Class])
        """
        return np.stack(self.replay_memory_state[i], axis=2)

    def get_state_next(self,i):
        """
        Gets a state_next  from a experience located on index i from inside the replay memory.

        :param  i :  int
                Index of the item (int)

        :return: state_next (np.array dtype=uint8 of shape input_shape [DQN Class])
        """
        return np.stack(self.replay_memory_state_next[i], axis=2)

    def get_action(self,i):
        """
        Gets a action  from a experience located on index i from inside the replay memory.

        :param  i :  int
                Index of the item (int)

        :return: action (np.int32)
        """
        return self.replay_memory_action[i]

    def get_reward(self,i):
        """
        Gets a reward from a experience located on index i from inside the replay memory.

        :param  i :  int
                Index of the item (int)

        :return: reward (np.float32)
        """
        return self.replay_memory_reward[i]

    def get_done(self,i):
        """
        Gets the done flag(if this experience is a terminal one) from a experience located on
        index i from inside the replay memory.

        :param  i :  int
                Index of the item (int)

        :return: done flag (np.float32)
        """
        return self.replay_memory_done[i]

    def state_next_save(self,i,name):
        """
        Function that saves the state as a gif on the disk.

        :param  i : int
                Index of the state to be saved(int)
        :param name: str
                Name of the file that will be saved on the disk.

        :return: nothing
        """
        folder_exists(self.path_save)
        img = self.get_state_next(i)
        n_frames = img.shape[2]/(self.input_depth)
        imageio.mimwrite(os.path.join(self.path_save, "{}{}.gif".format(name,i)),
                         np.split(img, n_frames, axis=2), fps=30)

    def state_save(self,i,name):
        """
        Function that saves the state_next as a gif on the disk.

        :param  i : int
                Index of the state_next to be saved(int)
        :param name: str
                Name of the file that will be saved on the disk.

        :return: nothing
        """
        folder_exists(self.path_save)
        img = self.get_state(i)
        n_frames = img.shape[2]/(self.input_depth)
        img = np.split(img, n_frames, axis=2)
        imageio.mimwrite(os.path.join(self.path_save, "{}{}.gif".format(name,i)),
                         img, fps=30)
