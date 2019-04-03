from abc import ABC, abstractmethod


class AbstractSession(ABC):
    @abstractmethod
    def epoch(self):
        pass

    @abstractmethod
    def step(self, samples_batch):
        pass
