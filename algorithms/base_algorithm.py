class BaseAlgorithm:
    """Base class providing metrics utilities."""

    def __init__(self):
        self._runtime = None
        self._memory_rss = None
        self._memory_uss = None

    def readfile(self, path, mode="r", encoding="utf-8"):
        """Return contents of a file."""
        with open(path, mode, encoding=encoding) as f:
            return f.read()

    def getruntime(self):
        return self._runtime

    def getmemoryrss(self):
        return self._memory_rss

    def getmemoryuss(self):
        return self._memory_uss

    def printresults(self):
        print(f"Runtime: {self._runtime}")
        print(f"Memory RSS: {self._memory_rss}")
        print(f"Memory USS: {self._memory_uss}")
