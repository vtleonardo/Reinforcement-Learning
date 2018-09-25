import numpy as np
from collections import deque
import imageio
import os

class ReplayMemory():
    def __init__(self,num_states_stored = 1000000,batch_size=32):
        self.path_save=os.path.join(os.path.dirname(os.path.realpath(__file__)),"Imagens")
        self.batch_size = batch_size
        self.num_states_stored = num_states_stored
        self._first_episode = True
        self._cont_idx = 0
        self.offset = 0
        self.state_idx = deque(maxlen=self.num_states_stored)
        self.replay_memory_state_zero = deque(maxlen=self.num_states_stored)
        self.replay_memory_action = deque(maxlen=self.num_states_stored)
        self.replay_memory_reward = deque(maxlen=self.num_states_stored)
        self.replay_memory_state_next = deque(maxlen=self.num_states_stored)
        self.replay_memory_done = deque(maxlen=self.num_states_stored)

        self.state_to_return = deque(maxlen=self.batch_size)
        self.action_to_return = deque(maxlen=self.batch_size)
        self.reward_to_return = deque(maxlen=self.batch_size)
        self.state_next_to_return = deque(maxlen=self.batch_size)
        self.done_to_return = deque(maxlen=self.batch_size)

    def append(self,state,action,reward,state_next,done):
        assert state.dtype == np.uint8,"State deve possuir tipo dtype=numpy.uint8"
        assert state_next.dtype == np.uint8, "State_next deve possuir tipo dtype=numpy.uint8"
        if self._first_episode:
            self._first_episode = False
            self.update_index()
            self._cont_idx = len(self.replay_memory_state_zero)
            self.state_idx.append(self._cont_idx)
            self.replay_memory_state_zero.append(np.copy(state))
        else:
            self.state_idx.append(-1)
        self.replay_memory_action.append(action)
        self.replay_memory_reward.append(reward)
        self.replay_memory_state_next.append(np.copy(state_next))
        self.replay_memory_done.append(done*1)
        #Caso state terminal, o próximo terá um estado inicial
        if done:
            self._first_episode=True

    def sample(self):
        idx = np.random.choice(len(self.replay_memory_done), self.batch_size)
        for i in idx:
            if (self.state_idx[i] == -1):
                self.state_to_return.append(self.replay_memory_state_next[i-1])
            else:
                self.state_to_return.append(self.replay_memory_state_zero[self.state_idx[i]])
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

    def update_index(self):
        if len(self.state_idx)>=self.num_states_stored:
            element = self.state_idx.popleft()
            if element != -1:
                self.replay_memory_state_zero.popleft()
                vect=np.array(self.state_idx)
                vect[vect != -1] -= 1
                self.state_idx = deque(vect,maxlen=self.num_states_stored)

    def size(self):
        return len(self.replay_memory_reward)
    def get_state(self,i):
        if (self.state_idx[i] == -1):
            return self.replay_memory_state_next[i - 1]
        else:
            return self.replay_memory_state_zero[self.state_idx[i]]
    def get_state_next(self,i):
        return self.replay_memory_state_next[i]
    def get_action(self,i):
        return self.replay_memory_action[i]
    def get_reward(self,i):
        return self.replay_memory_reward[i]
    def get_done(self,i):
        return self.replay_memory_done[i]
    def get_itens(self,i):
        return (self.get_state(i),
                self.get_action(i),
                self.get_reward(i),
                self.get_state_next(i),
                self.get_done(i))

    def state_next_save(self,i,path_save,name):
        self.folder_exists()
        imageio.mimsave(os.path.join(self.path_save, "{}{}.gif".format(name,i)),
                        np.rollaxis(self.get_state_next(i), 2, 0))

    def state_save(self,i,path_save,name):
        self.folder_exists()
        imageio.mimsave(os.path.join(self.path_save, "{}{}.gif".format(name,i)),
                        np.rollaxis(self.get_state(i), 2, 0))
    def folder_exists(self):
        if not (os.path.exists(self.path_save)):
            os.mkdir(self.path_save)