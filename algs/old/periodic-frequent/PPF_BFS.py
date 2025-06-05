import numpy as np
import time

class PPF_BFS:
    def __init__(self, file, minSup, maxPer, per_ratio, sep, output_file, num_cores = 1):
        self.file = file
        self.minSup = minSup
        self.maxPer = maxPer
        self.sep = sep
        self.per_ratio = per_ratio
        self.output_file = output_file
        self.num_cores = num_cores
        self.Patterns = {}
        
        
        
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
    
    def _getPerSup(self, arr):
        """
        This function takes the arr as input and returns locs as output

        :param arr: an array contains the items.
        :type arr: array
        :return: locs
        """
        copy = arr.copy()
        copy.add(self.max_tid)
        copy.add(0)
        copy = np.array(list(copy))
        copy = np.sort(copy)
        copy = np.diff(copy)

        locs = len(np.where(copy <= self.maxPer)[0])

        return locs   
    
    def process_item(self,k, v):
        if len(v) >= self.minSup:
            perSup = self._getPerSup(v)
            ratio = perSup / (len(v) + 1)
            if ratio >= self.per_ratio:
                self.Patterns[k] = [len(v), ratio]
            return k, v, ratio
        return None, None, None
 
    def _process_loop(self, cands, i):
        nCands = []
        for j in range(i+1, len(cands)):
            k1 = cands[i][0]
            k2 = cands[j][0]
            if k1[:-1] == k2[:-1] and k1[-1] != k2[-1]:
                nCand = k1.copy()
                nCand.append(k2[-1])
                v = cands[i][1].intersection(cands[j][1])
                if len(v) >= self.minSup:
                    perSup = self._getPerSup(v)
                    ratio = perSup / (len(v) + 1)
                    nCands.append([nCand, v])
                    if ratio >= self.per_ratio:
                        self.Patterns[tuple(nCand)] = [len(v), ratio]
            else:
                break
        return nCands
 
 
    def mine(self):
        self.start = time.time()
        items = self.read_file()
        
        cands = []
        nitems = {}
        
        for k, v in items.items():
            k, v, ratio = self.process_item(k, v)
            if k is not None:
                nitems[k] = v
                if ratio >= self.per_ratio:
                    self.Patterns[k] = [len(v), ratio]
                cands.append([[k], v])
                    
        while cands:
            nCands = []
            for i in range(len(cands)):
                for j in range(i+1, len(cands)):
                    k1 = cands[i][0]
                    k2 = cands[j][0]
                    if k1[:-1] == k2[:-1] and k1[-1] != k2[-1]:
                        nCand = k1.copy()
                        nCand.append(k2[-1])
                        v = cands[i][1].intersection(cands[j][1])
                        if len(v) >= self.minSup:
                            perSup = self._getPerSup(v)
                            ratio = perSup / (len(v) + 1)
                            nCands.append([nCand, v])
                            if ratio >= self.per_ratio:
                                self.Patterns[tuple(nCand)] = [len(v), ratio]
                    else:
                        break
            cands = nCands

        
        
        print("Number of patterns found: ", len(self.Patterns))
        print("Time taken: ", time.time()-self.start)
   
        
file = "/home/tarun/cuPAMI/datasets/Temporal_T10I4D100K.csv"
minSup = 10
maxPer = 5000
per_ratio = 1
sep = "\t"
output_file = "output.txt"

miner = PPF_BFS(file, minSup, maxPer, per_ratio, sep, output_file, 8)
miner.mine()