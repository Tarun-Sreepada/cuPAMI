import abstract
import kvikio
import cupy as cp

num_new_lines = cp.RawKernel(r'''
extern "C" __global__ void num_new_lines(const char *data, const unsigned long long int size, unsigned int *numLines) {
    unsigned long long int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;
    if (data[tid] == '\n') atomicAdd(&numLines[tid % 32], 1);
}
''', 'num_new_lines')
        
find_new_lines = cp.RawKernel(r'''
extern "C" __global__ void find_new_lines(const char *data, const unsigned long long int size, unsigned long long int *newline_indices) {
    unsigned long long int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;
    if (data[tid] == '\n') {
        unsigned long long int index = atomicAdd(&newline_indices[0], 1);
        newline_indices[index + 1] = tid;
    }
}
''', 'find_new_lines')


get_items_per_line = cp.RawKernel(r'''
                                  
extern "C" __global__ void get_items_per_line(const char *data,
                                const unsigned long long int *indexes, const int numLines, int *items_per_line, const char delimiter) {
    unsigned long long int lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx < numLines) {
        for (unsigned long long int i = indexes[lineIdx]; i < indexes[lineIdx + 1]; i++) {
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

//  (file_data, newline_indices, number_of_transactions, parsed_indices, raw_data, ord(self.delimiter)))
extern "C" __global__ void convert_char_file_to_int_file(
        const char *data, const unsigned long long int *indexes, const int numLines, const unsigned long long int *itemsPerLine, unsigned int *rawData, const char seperator) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numLines) return;
                            
    unsigned long long int start = indexes[tid];
    if (tid != 0) start++;
    unsigned long long int end = indexes[tid + 1];
    char buffer[32];
    int bufferIndex = 0;

    unsigned long long int j = itemsPerLine[tid];
    for (unsigned long long int k = start; k < end; k++) {
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
    def __init__(self, file, delimiter = ',', warmup = False):
        super().__init__(file, delimiter)
        self.warmup = warmup
    def read(self):
        
        manual = 0
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
        print("File size: ", file_size)
        
        
        file_data = cp.empty(file_size, dtype=cp.uint8)
        manual += file_size
        
        with kvikio.CuFile(self.file, "r") as f:
            f.raw_read(file_data)
            
        # count number of transactions
        numLines = cp.zeros(32, dtype=cp.uint32)
        num_new_lines(
                (file_size // self.block_size + 1,),
                (self.block_size,),
                (file_data, file_size, numLines)
            )

        cp.cuda.Device().synchronize()
        number_of_transactions = int(numLines.sum())
        print("Number of transactions: ", number_of_transactions)
        
        # get indices of newline characters
        newline_indices = cp.zeros(number_of_transactions + 1, dtype=cp.uint64)
        manual += newline_indices.nbytes
        find_new_lines(
            (file_size // self.block_size + 1,),
            (self.block_size,),
            (file_data, file_size, newline_indices)
        )
        # set the first element to 0
        newline_indices[0] = 0
        # sort newline_indices
        newline_indices = cp.sort(newline_indices).astype(cp.uint64)
        
        
        # # total number of items
        items_per_line = cp.zeros(number_of_transactions + 1, dtype=cp.int32)
        manual += items_per_line.nbytes
        

        get_items_per_line((number_of_transactions//self.block_size + 1,), (self.block_size,), (file_data, 
                                                        newline_indices, number_of_transactions, items_per_line, cp.uint8(ord(self.delimiter))))
        # cp.cuda.Device().synchronize()
        # # add 1 to all elements
        items_per_line = items_per_line + 1
        items_per_line[0] = 0
        
        parsed_indices = cp.cumsum(items_per_line).astype(cp.uint64)
        manual += parsed_indices.nbytes
        number_of_items = parsed_indices[-1].get()
        
        raw_data = cp.zeros(number_of_items, dtype=cp.uint32)
        manual += raw_data.nbytes
        
        convert_char_file_to_int_file((number_of_transactions//self.block_size + 1,), (self.block_size,), 
                                      (file_data, newline_indices, number_of_transactions, parsed_indices, raw_data, cp.uint8(ord(self.delimiter))))
        
        
        
        self.runtime = abstract.time.time() - start
        
        # get memory usage
        end_memory = cp.get_default_memory_pool().used_bytes()
        
        if end_memory < start_memory or end_memory == start_memory:
            # manually calculate memory usage
            self.custom_memory["gpu"] = manual
        else: 
            self.custom_memory["gpu"] = end_memory - start_memory
            
        print("Last element: ", raw_data[-1])
        
        return raw_data, parsed_indices
        
        
    
if __name__ == "__main__":
    
    cur_dir  = abstract.os.path.dirname(__file__)
    file = "../../datasets/synthetic/transactional/triangle_4096M.csv"
    
    file = abstract.os.path.join(cur_dir, file)
    
    obj = gpu_read(file)
    obj.read()
    print(obj.get_runtime())
    print(abstract.bytes_to_mb(obj.get_memory()), "MB")
    print(abstract.bytes_to_mb(obj.get_custom_memory()["gpu"]), "MB")