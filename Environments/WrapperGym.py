################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################
import cv2
import numpy as np
from collections import deque
import gym
from gym import spaces
from Environments.Base import Base


class WrapperGym(Base):
    """
    Wrapper for gym environments to match all methods required by the DQN algorithm.

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
    def __init__(self,env):
        """
        Creates the object

        :param
        env: str
            Name of the gym environment
        """
        self.env=wrap_deepmind(gym.make(env))
        self.is_to_render = False

    def getName(self):
        """
        Function that gets the environment's name

        :return: str
            String with the environment's name
        """
        return self.env.spec.id

    def render(self, is_to_render=False):
        """
        Controls if it's to display the environment on the screen to the user.

        :param is_to_render : bool (Default : False)
                Flag that controls if it's to display the environment on the screen to the user.
        :return:
            nothing
        """
        self.is_to_render = is_to_render

    def reset(self):
        """
        Resets the environment ant its variables and send back the initial state.

        :return: numpy.array [dtype=uint8] with shape (input_shape from you RL_algorithm)
            initial state (frame image [np.array of dtype=uint8]) of the environment
        """
        return self.env.reset()

    def step(self,action):
        """
        Take an action in the environment and return the environment's feedback.
        :param
        action: int
            Action to be taken in the environment
        :return: tuple compose of:
            the reward : float
            the next state : np.array of dtype=uint8 with shape (input_shape from you RL_algorithm)
            done : bool (flag that tells us if the episode has done)
            info : no type defined
        """
        if self.is_to_render:
            self.env.render()
        return self.env.step(action)

    def numberOfActions(self):
        """
        Function that returns the number os possible actions for this environment.

        :return: int
            The number of possible actions for this environment
        """
        return self.env.action_space.n
    def set_seed(self,seed):
        """
        Function that sets the random seed of the environment.

        :param
        seed : int
            Seed to set the random function of this environment

        :return:
            nothing
        """
        self.env.seed(seed)

    def action_meanings(self):
        """
        Function that gets the meaning (name) of each action inside the environment.

        :param nothing

        :return: list of str
            Name of each action available inside the environment
        """
        return self.env.unwrapped.get_action_meanings()

    def close(self):
        """
        Function that closes the current environment.

        :param nothing

        :return: nothing
        """
        self.env.close()



################################################################################################################
# This wrapper are based on dqn paper (nature) and on the article:                                             #
# Speeding up DQN on PyTorch: how to solve Pong in 30 minutes (link below)                                     #
# https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55               #
################################################################################################################

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def _reset(self):
        return _process_frame84(self.env.reset())

class ClippedRewardsWrapper(gym.Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, np.sign(reward), done, info

def wrap_deepmind_ram(env):
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClippedRewardsWrapper(env)
    return env

def wrap_deepmind(env):
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ClippedRewardsWrapper(env)
    return env