import abstract
import sys

class python_read(abstract.AbstractRead):
    def read(self):
        start = abstract.time.time()
        with open(self.file) as f:
            file = [list(map(int, line.strip().split(self.delimiter))) for line in f]
            
        self.runtime = abstract.time.time() - start
        
        # get size of file in memory
        file_mem = sum([sys.getsizeof(row) for row in file])
        
        self.custom_memory["cpu"] = file_mem
        
        return file
    
if __name__ == "__main__":
    
    cur_dir  = abstract.os.path.dirname(__file__)
    file = "../../datasets/synthetic/transactional/square_1G.csv"
    
    file = abstract.os.path.join(cur_dir, file)
    
    obj = python_read(file, ",")
    obj.read()
    print(obj.getRuntime())
    print(abstract.bytes_to_mb(obj.getMemory()), "MB")
    print(abstract.bytes_to_mb(obj.getCustomMemory()["cpu"]), "MB")