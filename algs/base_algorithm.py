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

    def readFile(self, path, *, mode="r", encoding="utf-8"):
        """Read and return the contents of *path*.

        Parameters
        ----------
        path : str
            Path to the file to read.
        mode : str, optional
            File mode used when opening. Defaults to ``"r"``.
        encoding : str, optional
            Encoding to decode the file with. Defaults to ``"utf-8"``.
        """
        with open(path, mode, encoding=encoding) as f:
            return f.read()
=======
