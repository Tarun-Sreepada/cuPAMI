from abc import ABC, abstractmethod
import psutil
import os
import time

def bytes_to_mb(bytes):
    return bytes / 1024 / 1024

class AbstractRead(ABC):
    def __init__(self, file, delimiter = ','):
        self.file = file
        self.delimiter = delimiter
        self.num_cols = 0
        self.custom_memory = {}
        self.runtime = None  # Placeholder for runtime
        if self.file.endswith(".csv"):
            self.get_num_cols()

    def get_num_cols(self):
        """
        Reads the first line of the file and calculates the number of columns based on the delimiter.
        """
        with open(self.file, 'r') as f:
            line = f.readline()
            self.num_cols = len(line.split(self.delimiter))

    @abstractmethod
    def read(self):
        """
        Abstract method for reading the file. Must be implemented by subclasses.
        """
        pass

    def get_runtime(self):
        """
        Returns the runtime of the file reading process.
        """
        return self.runtime

    def get_memory(self):
        """
        Returns the current memory usage of the process in bytes.
        """
        pid = os.getpid()
        rss = psutil.Process(pid).memory_info().rss
        return rss

    def get_custom_memory(self):
        """
        Returns any custom memory usage information tracked by the implementation.
        """
        return self.custom_memory
