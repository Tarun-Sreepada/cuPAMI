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
            
    def mine(self):
        start = time.time()
        self.read_file()
        
        # sort in descending order of support
        self.indices_dict = {k: v for k, v in sorted(self.indices_dict.items(), key=lambda item: len(item[1]), reverse=True)}
        
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
            
        self.runtime = time.time() - start
            
    def printResults(self):
        print("Number of patterns: ", len(self.Patterns))
        print("Runtime: ", self.runtime)
    
    def getPatterns(self):
        return self.Patterns
    
    def getRuntime(self):
        return self.runtime
       
        
if __name__ == '__main__':
    
     # obj = cuFPMiner_bit("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.parquet", 20, '\t', 'parquet', 'device')
    # obj.mine()
    # obj.printResults()
    
    # apriori = Apriori("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.parquet", 20, '\t', 'parquet')
    # apriori.mine()
    # apriori.printResults()
    
    apriori = Apriori("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.csv", 1000, '\t', 'csv')
    apriori.mine()
    apriori.printResults()