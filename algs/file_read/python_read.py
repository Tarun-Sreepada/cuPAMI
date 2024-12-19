import abstract
import sys

class python_read(abstract.AbstractRead):
    def read(self):
        start = abstract.time.time()

        # Use readlines and splitlines for faster processing
        with open(self.file, 'r') as f:
            lines = f.readlines()

        # Efficient processing of lines
        file = [
            [int(value) for value in line.split(self.delimiter)]
            for line in lines
        ]

        self.runtime = abstract.time.time() - start

        # Calculate memory usage more efficiently
        file_mem = sum(sys.getsizeof(row) for row in file) + sys.getsizeof(file)

        self.custom_memory["cpu"] = file_mem

        return file
    
if __name__ == "__main__":
    
    cur_dir  = abstract.os.path.dirname(__file__)
    file = "../../datasets/synthetic/transactional/square_1100M.csv"
    
    file = abstract.os.path.join(cur_dir, file)
    
    obj = python_read(file, ",")
    obj.read()
    print(obj.get_runtime())
    print(abstract.bytes_to_mb(obj.get_memory()), "MB")
    print(abstract.bytes_to_mb(obj.get_custom_memory()["cpu"]), "MB")