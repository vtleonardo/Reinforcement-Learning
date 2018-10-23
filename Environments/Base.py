################################################################################################################
# Created by Leonardo Viana Teixeira at 17/10/2018                                                             #
################################################################################################################

from abc import ABC,abstractmethod


class Base(ABC):
    """
    Base class for all environments specifying what methods they must have!
    All environments must inherit this class.
    """
    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, is_to_render=False):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def numberOfActions(self):
        pass
    @abstractmethod
    def set_seed(self, seed):
        pass

    @abstractmethod
    def close(self):
        pass
