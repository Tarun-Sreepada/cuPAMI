import abstract
import cudf
import pandas as pd

class cudf_read(abstract.AbstractRead):
    def __init__(self, file, delimiter=',', file_type="csv"):
        super().__init__(file, delimiter)
        self.custom_memory = {}
        self.file_type = file_type
        
    
    def read(self):
        start = abstract.time.time()
        # df = cudf.read_csv(self.file, delimiter=self.delimiter, header=None)
        if self.file_type == "csv":
            df = cudf.read_csv(self.file, delimiter=self.delimiter, header=None)
        elif self.file_type == "parquet":
            df = cudf.read_parquet(self.file)
        self.runtime = abstract.time.time() - start
        
        # get gpu memory usage
        self.custom_memory["gpu"] = df.memory_usage().sum()
        
        return df
    
if __name__ == "__main__":
    
    cur_dir  = abstract.os.path.dirname(__file__)
    file = "../../datasets/synthetic/transactional/square_1500M.csv"
    
    file = abstract.os.path.join(cur_dir, file)
    
    obj = cudf_read(file, ",", "csv")
    df = obj.read()
    print(obj.get_runtime())
    print(abstract.bytes_to_mb(obj.get_memory()), "MB")
    print(abstract.bytes_to_mb(obj.get_custom_memory()["gpu"]), "MB")
    
