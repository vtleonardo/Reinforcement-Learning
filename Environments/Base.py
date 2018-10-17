from abc import ABC,abstractmethod

"""
Base class for all environments especifing what methods they must have! All environments must inherit this class.
"""
class Base(ABC):

    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
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
