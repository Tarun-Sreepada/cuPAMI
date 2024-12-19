import abstract
import pandas as pd

class pandas_read(abstract.AbstractRead):
    def read(self):
        start = abstract.time.time()
        df = pd.read_csv(self.file, delimiter=self.delimiter, header=None)
        self.runtime = abstract.time.time() - start
        
        self.custom_memory["cpu"] = df.memory_usage(deep=True).sum()
        
        return df
    
if __name__ == "__main__":
    
    cur_dir  = abstract.os.path.dirname(__file__)
    file = "../../datasets/synthetic/transactional/square_1G.csv"
    
    file = abstract.os.path.join(cur_dir, file)
    
    obj = pandas_read(file, ",")
    obj.read()
    print(obj.getRuntime())
    print(abstract.bytes_to_mb(obj.getMemory()), "MB")
    print(abstract.bytes_to_mb(obj.getCustomMemory()["cpu"]), "MB")