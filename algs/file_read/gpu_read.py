import abstract
import kvikio
import cupy as cp

get_items_per_line = cp.RawKernel(r'''
                                  
extern "C" __global__ void get_items_per_line(const char *data,
                                const int *indexes, const int numLines, int *items_per_line, const int delimiter) {
    int lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx < numLines) {
        for (int i = indexes[lineIdx]; i < indexes[lineIdx + 1]; i++) {
            if (data[i] == delimiter) {
                items_per_line[lineIdx + 1]++;
            }
        }
        
    }
}
''', 'get_items_per_line')


convert_char_file_to_int_file = cp.RawKernel(r'''
                            
__device__ int my_atoi(const char *str) {
    int result = 0;
    int sign = 1;
    int i = 0;

    // Skip any leading spaces
    while (str[i] == ' ') {
        i++;
    }

    // If the number is negative, update the sign and start position
    if (str[i] == '-') {
        sign = -1;
        i++;
    }

    // Iterate over each character and construct the integer
    for (; str[i] != '\0'; ++i) {
        if (str[i] < '0' || str[i] > '9') {
            break; // Break if a non-numeric character is encountered
        }
        result = result * 10 + (str[i] - '0');
    }

    return sign * result;
}


extern "C" __global__ void convert_char_file_to_int_file(
        const char *data, const int *indexes, const int numLines, const int *itemsPerLine, int *rawData, const int seperator) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numLines) return;
                            
    int start = indexes[tid];
    int end = indexes[tid + 1];
    char buffer[32];
    int bufferIndex = 0;

    int j = itemsPerLine[tid];
    start++;
    for (int k = start; k < end; k++) {
        if (data[k] != seperator) {
            buffer[bufferIndex++] = data[k];
        }
        else {
            buffer[bufferIndex] = '\0';
            rawData[j++] = my_atoi(buffer);
            bufferIndex = 0;
        }
    }
    rawData[j] = my_atoi(buffer);
                            
}

''', 'convert_char_file_to_int_file')

class gpu_read(abstract.AbstractRead):
    def __init__(self, file, delimiter, warmup=False):
        super().__init__(file, delimiter)
        self.warmup = warmup
    def read(self):
        self.block_size = 256
        
        #warm up kvikio
        if self.warmup:
            temp_start = abstract.time.time()
            with kvikio.CuFile(self.file, "r") as f:
                pass
            temp_end = abstract.time.time()
            print("Warm up time for kvikio: ", temp_end - temp_start)
        
        # cupy memory pool context so we can subtract the memory allocated by cupy'
        start_memory = cp.get_default_memory_pool().used_bytes()

        start = abstract.time.time()
        
        
        # read data
        file_size = abstract.os.path.getsize(self.file)
        
        
        file_data = cp.empty(file_size, dtype=cp.uint8)
        with kvikio.CuFile(self.file, "r") as f:
            f.read(file_data)
            
        # count number of transactions
        number_of_transactions = cp.count_nonzero(file_data == ord('\n')).get()
        
        # get indices of newline characters
        newline_indices = cp.where(file_data == ord('\n'))[0]
        
        # add 0 to the front of newline_indices and size of file to the end of newline_indices
        newline_indices = cp.concatenate([cp.array([0]), newline_indices])
        newline_indices = cp.concatenate([newline_indices, cp.array([file_size])])
        newline_indices = cp.sort(newline_indices).astype(cp.int32)
        
        # total number of items
        items_per_line = cp.zeros(number_of_transactions + 1, dtype=cp.int32)
        get_items_per_line((number_of_transactions//self.block_size + 1,), (self.block_size,), (file_data, 
                                                        newline_indices, number_of_transactions, items_per_line, ord(self.delimiter)))
        cp.cuda.Device().synchronize()
        # add 1 to all elements
        items_per_line = items_per_line + 1
        items_per_line[0] = 0
        
        parsed_indices = cp.cumsum(items_per_line).astype(cp.int32)
        number_of_items = parsed_indices[-1].astype(cp.int32).get()
        
        raw_data = cp.zeros(number_of_items, dtype=cp.int32)
        convert_char_file_to_int_file((number_of_transactions//self.block_size + 1,), (self.block_size,), 
                                      (file_data, newline_indices, number_of_transactions, parsed_indices, raw_data, ord(self.delimiter)))
        
        self.number_of_transactions = number_of_transactions
        
        
        self.runtime = abstract.time.time() - start
        
        # get memory usage
        end_memory = cp.get_default_memory_pool().used_bytes()
        
        self.custom_memory["gpu"] = end_memory - start_memory
        
        return raw_data, parsed_indices
        
        
    
if __name__ == "__main__":
    
    cur_dir  = abstract.os.path.dirname(__file__)
    file = "../../datasets/synthetic/transactional/square_1G.csv"
    obj = gpu_read(file, ",", warmup=True)
    file = abstract.os.path.join(cur_dir, file)
    
    obj = gpu_read(file, ",")
    obj.read()
    print(obj.getRuntime())
    print(abstract.bytes_to_mb(obj.getMemory()), "MB")
    print(abstract.bytes_to_mb(obj.getCustomMemory()["gpu"]), "MB")