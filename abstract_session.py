from abc import ABC, abstractmethod


class AbstractSession(ABC):
    @abstractmethod
    def epoch(self):
        pass

    @abstractmethod
    def step(self, samples_batch):
        pass

    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            print("name of {} unregistered".format(self))
            return None
