import numpy as np
import time
from numba import njit
import multiprocessing as mp

@njit
def _getPerSup(arr, max_tid, maxPer):
    """
    This function takes the arr as input and returns locs as output

    :param arr: an array contains the items.
    :type arr: array
    :return: locs
    """
    copy = arr.copy()
    copy.add(max_tid)
    copy.add(0)
    copy = np.array(list(copy))
    copy = np.sort(copy)
    copy = np.diff(copy)

    locs = len(np.where(copy <= maxPer)[0])

    return locs

def process_candidates_subset(cand_subset, minSup, max_tid, maxPer, per_ratio):
    """
    Process a given subset of candidates (not necessarily contiguous in the original list).
    """
    newCands = []
    patterns_partial = {}
    length = len(cand_subset)
    for i in range(length):
        for j in range(i+1, length):
            intersection = cand_subset[i][1].intersection(cand_subset[j][1])
            if len(intersection) >= minSup:
                perSup = _getPerSup(intersection, max_tid, maxPer)
                ratio = perSup / (len(intersection) + 1)
                nCand = cand_subset[i][0] + cand_subset[j][0]
                newCands.append([nCand, intersection])
                if ratio >= per_ratio:
                    patterns_partial[tuple(set(nCand))] = [len(intersection), ratio]
    return newCands, patterns_partial

class PPF_DFS:
    def __init__(self, file, minSup, maxPer, per_ratio, sep, output_file, num_cores = 1):
        self.file = file
        self.minSup = minSup
        self.maxPer = maxPer
        self.sep = sep
        self.per_ratio = per_ratio
        self.output_file = output_file
        self.num_cores = num_cores
        self.Patterns = mp.Manager().dict()
        
    def read_file(self):
        with open(self.file, 'r') as f:
            lines = f.readlines()
        file = [x.split(self.sep) for x in lines]
        file = [[int(x) for x in y] for y in file]
        
        items = {}
        max_tid = 0
        for i in range(len(file)):
            tid = file[i][0]
            max_tid = max(max_tid, tid)
            for j in range(1, len(file[i])):
                if file[i][j] not in items:
                    items[file[i][j]] = [tid]
                items[file[i][j]].append(tid)
                
        self.max_tid = max_tid
        items = {k: set(v) for k, v in items.items() if len(v) >= self.minSup}
        
        items = dict(sorted(items.items(), key=lambda x: len(x[1]), reverse=True))
        
        return items
    
    def process_item(self,k, v):
        if len(v) >= self.minSup:
            perSup = _getPerSup(v, self.max_tid, self.maxPer)
            ratio = perSup / (len(v) + 1)
            if ratio >= self.per_ratio:
                self.Patterns[k] = [len(v), ratio]
            return k, v, ratio
        return None, None, None
 
    def __recursive_single(self, cands):
        
        for i in range(len(cands)):
            newCands = []
            for j in range(i+1, len(cands)):
                intersection = cands[i][1].intersection(cands[j][1])
                if len(intersection) >= self.minSup:
                    perSup = _getPerSup(intersection, self.max_tid, self.maxPer)
                    ratio = perSup / (len(intersection) + 1)
                    nCand = cands[i][0] + cands[j][0]
                    newCands.append([nCand, intersection])
                    if ratio >= self.per_ratio:
                        self.Patterns[tuple(set(nCand))] = [len(intersection), ratio]
                
            if newCands:
                self.__recursive_single(newCands)
                
    def mine(self):
        self.start = time.time()
        items = self.read_file()
        
        cands = []
        
        for k, v in items.items():
            k, v, ratio = self.process_item(k, v)
            if k is not None:
                if ratio >= self.per_ratio:
                    self.Patterns[k] = [len(v), ratio]
                cands.append([[k], v])
                    
        self.__recursive_single(cands)
        
        print("Number of patterns found: ", len(self.Patterns))
        print("Time taken: ", time.time()-self.start)
        
   
        
file = "/home/tarun/cuPAMI/datasets/Temporal_T10I4D100K.csv"
minSup = 5
maxPer = 5000
per_ratio = 1
sep = "\t"
output_file = "output.txt"

miner = PPF_DFS(file, minSup, maxPer, per_ratio, sep, output_file, 8)
miner.mine()