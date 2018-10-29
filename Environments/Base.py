################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################

from abc import ABC,abstractmethod


class Base(ABC):
    """
    Base class for all environments specifying what methods they must have!
    All environments must inherit this class and therefore it have the methods:
        getName: returns a string with the environment's name
        render:  make the environment's rendering
        reset: resets the environment and return the state
        step: does an action on the environment and returns its consequences
                    (the reward, the next state, if the episode has done, and other infos).
        numberOfActions: returns the number of actions possible on this environment.
        set_seed : set the random seed of the environment
        close : close the environment

    """
    @abstractmethod
    def getName(self):
        """
        Function that gets the environment's name

        :return: str
            String with the environment's name
        """
        pass

    @abstractmethod
    def render(self, is_to_render=False):
        """
        Controls if it's to display the environment on the screen to the user.

        :param is_to_render : bool (Default : False)
                Flag that controls if it's to display the environment on the screen to the user.
        :return:
            nothing
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the environment ant its variables and send back the initial state.

        :return: numpy.array [dtype=uint8] with shape (input_shape from you RL_algorithm)
            initial state (frame image [np.array of dtype=uint8]) of the environment
        """
        pass

    @abstractmethod
    def step(self, action):
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
        pass

    @abstractmethod
    def numberOfActions(self):
        """
         Function that returns the number os possible actions for this environment.

         :return: int
             The number of possible actions for this environment
         """
        pass
    @abstractmethod
    def set_seed(self, seed):
        """
        Function that sets the random seed of the environment.

        :param
        seed : int
            Seed to set the random function of this environment

        :return:
            nothing
        """
        pass

    @abstractmethod
    def action_meanings(self):
        """
        Function that gets the meaning (name) of each action inside the environment.

        :param nothing

        :return: list of str
            Name of each action available inside the environment
        """
        pass

    @abstractmethod
    def close(self):
        """
        Function that closes the current environment.

        :param nothing

        :return: nothing
        """
        pass
