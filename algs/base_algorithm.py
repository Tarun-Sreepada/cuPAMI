from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Base class for algorithm implementations.

    Subclasses should set ``runtime`` (seconds), ``memoryRSS`` (bytes) and
    ``memoryUSS`` (bytes) during execution.
    """

    def __init__(self) -> None:
        self.runtime = None
        self.memoryRSS = None
        self.memoryUSS = None

    @abstractmethod
    def mine(self) -> None:
        """Run the algorithm."""
        pass

    def getRuntime(self):
        """Return the runtime of the last execution."""
        return self.runtime

    def getMemoryRSS(self):
        """Return the resident set size memory usage."""
        return self.memoryRSS

    def getMemoryUSS(self):
        """Return the unique set size memory usage."""
        return self.memoryUSS
