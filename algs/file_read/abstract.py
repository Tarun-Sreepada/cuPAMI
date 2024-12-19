import psutil
import os
import time

def bytes_to_mb(bytes):
    return bytes / 1024 / 1024

class AbstractRead:
    def __init__(self, file, delimiter):
        self.file = file
        self.delimiter = delimiter
        self.num_cols = 0
        self.custom_memory = {}
        self.get_num_cols()
        
    def get_num_cols(self):
        # open the file read the first line and get the number of columns
        with open(self.file) as f:
            line = f.readline()
            self.num_cols = len(line.split(self.delimiter))
        
    def read(self):
        pass
    
    def getRuntime(self):
        return self.runtime
    
    def getMemory(self):
        pid = os.getpid()
        rss = psutil.Process(pid).memory_info().rss
        return rss
    
    def getCustomMemory(self):
        return self.custom_memory