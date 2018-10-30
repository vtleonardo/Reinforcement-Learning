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
    This class only stores a given state once and after that links its position to the next experience,
    thus saving memory resources. To understand the previous statement, consider st_0 and s_t+1 as a state and
    a state_next coming from an experience of time step t=0 and s_t+1 and s_t+2 coming from the experience at
    the time step t=+1. S_t+1 in the first experience will be the state resulted from the agent taking an action
    in the environment (state_next), however in the second, s_t+1 will be the state and the s_t+2 will
    be state_next. Thus, we need to store s_t+1 only in the first experience, in the second we only need to
    link its position (inside the state_next deque).

    """
    def __init__(self,num_states_stored=1000000,batch_size=32,path_save=""):
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
        self._first_state = True
        # Aux variable that keeps track of the length of state_zero
        self._cont_idx = 0
        self.state_idx = deque(maxlen=self.num_states_stored)
        # DEQUE with fixed length (num_states_stored) for each item inside an experience
        #Deque that stores the initial states of each episode
        self.replay_memory_state_initial = deque(maxlen=self.num_states_stored)
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

        If an experience comes from an initial state, we need to store the state and state_next variables,
        otherwise we only need to store the state_next variable and link state_next[-1] (from previous time
        step) as being the state variable. The state_idx is an auxiliary deque that receives:
        -1 if the experience is from a first state or
        the index where the initial state will be stored inside the memory_state_initial deque.

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
        # Calling the function that checks and manually pop an element in case state_idx is full
        self.update_index()
        if self._first_state:
            self._first_state = False
            self._cont_idx = len(self.replay_memory_state_initial)
            self.state_idx.append(self._cont_idx)
            self.replay_memory_state_initial.append(np.copy(state))
        else:
            self.state_idx.append(-1)
        self.replay_memory_action.append(action)
        self.replay_memory_reward.append(reward)
        self.replay_memory_state_next.append(np.copy(state_next))
        self.replay_memory_done.append(done*1)
        # If this is a terminal state, the next experience will be a initial state (first episode)
        if done:
            self._first_state=True

    def update_index(self):
        """
        This method pop out an element from state_idx in case is full. If the element in question is different
        from -1, meaning that corresponds to an index to an initial state, we also pop out the element from
        that index inside memory_state_initial and refresh the rest of the indexes.

        :return: nothing
        """
        if len(self.state_idx)>=self.num_states_stored:
            element = self.state_idx.popleft()
            if element != -1:
                self.replay_memory_state_initial.popleft()
                vect=np.array(self.state_idx)
                vect[vect != -1] -= 1
                self.state_idx = deque(vect,maxlen=self.num_states_stored)

    def sample(self):
        """
        Method that samples uniformly N (from size batch_size) elements from the replay memory.

        :return: tuple compose of:
            state : N elements of in a form of np.array where each element has a type of np.uint8

            action : N elements of in a form of np.array where each element has a type of np.int32

            reward : N elements of in a form of np.array where each element has a type of np.float32

            state_next : N elements of in a form of np.array where each element has a type of np.uint8

            done : N elements of in a form of np.array where each element has a type of np.float32
                    (flag(in float type) if this experience is an terminal one)
        """
        # Sampling N indexes of elements uniformly
        idx = np.random.choice(len(self.replay_memory_done), self.batch_size)
        for i in idx:
            if (self.state_idx[i] == -1):
                # If state_idx is -1 we can use as state the previous state_next
                self.state_to_return.append(self.replay_memory_state_next[i-1])
            else:
                # If state_idx is different from -1 this experience is an initial one thus we have
                # to get the correspondent initial state from replay memory initial state
                self.state_to_return.append(self.replay_memory_state_initial[self.state_idx[i]])
            self.action_to_return.append(self.replay_memory_action[i])
            self.reward_to_return.append(self.replay_memory_reward[i])
            self.state_next_to_return.append(self.replay_memory_state_next[i])
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
        if (self.state_idx[i] == -1):
            return self.replay_memory_state_next[i - 1]
        else:
            return self.replay_memory_state_initial[self.state_idx[i]]

    def get_state_next(self,i):
        """
        Gets a state_next  from a experience located on index i from inside the replay memory.

        :param  i :  int
                Index of the item (int)

        :return: state_next (np.array dtype=uint8 of shape input_shape [DQN Class])
        """
        return self.replay_memory_state_next[i]

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
        imageio.mimsave(os.path.join(self.path_save, "{}{}.gif".format(name,i)),
                        np.rollaxis(self.get_state_next(i), 2, 0))

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
        imageio.mimsave(os.path.join(self.path_save, "{}{}.gif".format(name,i)),
                        np.rollaxis(self.get_state(i), 2, 0))
