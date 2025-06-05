import pandas as pd
import numpy as np
import time

class Apriori:
    def __init__(self, file, minSup, sep, file_type):
        self.file = file
        self.minSup = minSup
        self.sep = sep
        self.file_type = file_type
        self.block_size = 512
        self.Patterns = []
    
    def read_parquet(self):
        # Read the Parquet file
        df = pd.read_parquet(self.file)
        
        # Melt the DataFrame to reshape it into long-form
        melted_df = df.reset_index().melt(id_vars="index", var_name="column", value_name="value")
        
        # Group by value and collect indices as sets
        self.indices_dict = (
            melted_df.groupby("value")[["index", "column"]]
            .apply(lambda x: set(x["index"]))
            .to_dict()
        )

    def read_csv(self):
        # Read the CSV file
        lines = []
        with open(self.file, 'r') as f:
            lines = f.readlines()
            
        lines = [line.strip().split(self.sep) for line in lines]
        self.indices_dict = {}
        for i, line in enumerate(lines):
            for item in line:
                if item not in self.indices_dict:
                    self.indices_dict[item] = set()
                self.indices_dict[item].add(i)      
        
    def read_file(self):
        if self.file_type == 'parquet':
            self.read_parquet()
        elif self.file_type == 'csv':
            self.read_csv()
            
    def mine(self, memory_save=False):
        start = time.time()
        self.read_file()
        
        # sort in descending order of support
        self.indices_dict = {k: v for k, v in sorted(self.indices_dict.items(), key=lambda item: len(item[1]), reverse=True)}
        
        if memory_save == False:
            cands = [[[k],v] for k,v in self.indices_dict.items() if len(v) >= self.minSup]
            self.Patterns = [[cand, len(cand)] for cand in cands]
            while len(cands) > 1:
                nCands = []
                for i in range(len(cands)):
                    cand_i = cands[i][0]
                    for j in range(i+1, len(cands)):
                        cand_j = cands[j][0]
                        # if the first k-1 elements are the same and the last element is different
                        if cand_i[:-1] == cand_j[:-1] and cand_i[-1] != cand_j[-1]:
                            intersection = cands[i][-1] & cands[j][-1]
                            if len(intersection) >= self.minSup:
                                nCand = cand_i + [cand_j[-1]]
                                nCands.append([nCand, intersection])
                                self.Patterns.append([nCand, len(intersection)])
                        else:
                            break
                cands = nCands
        else:
            cands = [[k] for k,v in self.indices_dict.items() if len(v) >= self.minSup]
            self.Patterns = [[cand, len(self.indices_dict[cand[0]])] for cand in cands]
            while len(cands) > 1:
                nCands = []
                for i in range(len(cands)):
                    cand_i = cands[i]
                    for j in range(i+1, len(cands)):
                        cand_j = cands[j]
                        if cand_i[:-1] == cand_j[:-1] and cand_i[-1] != cand_j[-1]:
                            intersection = self.indices_dict[cand_j[-1]]
                            for k in range(0, len(cand_i)):
                                intersection = intersection & self.indices_dict[cand_i[k]]
                            if len(intersection) >= self.minSup:
                                nCand = cand_i + [cand_j[-1]]
                                nCands.append(nCand)
                                self.Patterns.append([nCand, len(intersection)])
                        else:
                            break
                cands = nCands
            
        self.runtime = time.time() - start
            
    def printResults(self):
        print("Number of patterns: ", len(self.Patterns))
        print("Runtime: ", self.runtime)
    
    def getPatterns(self):
        return self.Patterns
    
    def getRuntime(self):
        return self.runtime
    
    def save_patterns(self, output_file: str, separator: str = '\t') -> None:
        """Save mined patterns to a file."""
        with open(output_file, 'w') as f:
            for pattern, count in self.patterns.items():
                f.write(f"{separator.join(pattern)}:{count}\n")
       
        
if __name__ == '__main__':
    
    #  obj = cuFPMiner_bit("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.parquet", 20, '\t', 'parquet', 'device')
    # obj.mine()
    # obj.printResults()
    
    apriori = Apriori("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.parquet", 10, '\t', 'parquet')
    apriori.mine()
    apriori.printResults()
    
    # apriori = Apriori("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.parquet", 20, '\t', 'parquet')
    # apriori.mine(memory_save=True)
    # apriori.printResults()
    
    # file = "/home/tarun/cuPAMI/datasets/Transactional_pumsb.csv"
    # sep = '\t'
    # minSup = 38000
    # outFile = "patterns.txt"
    
    
    # # apriori = Apriori("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.csv", 1000, '\t', 'csv')
    # apriori = Apriori(file, minSup, sep, 'csv')
    # apriori.mine(memory_save=True)
    # apriori.printResults()
    # apriori.getPatterns()