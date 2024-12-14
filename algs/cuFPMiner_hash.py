import cupy as cp
import numpy as np
import time
from collections import Counter
import kvikio
import cudf
import os


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
    if (tid !=0 ) start++;
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


hash_table = cp.RawKernel(r'''
                          
__device__ unsigned int pcg_hash(int input)
{
    int state = input * 747796405u + 2891336453u;
    int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash function
__device__ unsigned int hashFunction(int key, int tableSize)
{
    return pcg_hash(key) % tableSize;
}
                          
                          
extern "C" 
__global__ void hash_table(const int *data, const int *indexes, int numLines, int *hash_table, const int load_factor, int *supports) 
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numLines) return;
    
    int start = indexes[tid];
    int end = indexes[tid + 1];
    int bucket_size = end - start;
    bucket_size *= load_factor;
    
    
    int insert_start = start * load_factor;
    
    for (int i = start; i < end; i++) {
        int item = data[i];
        atomicAdd(&supports[item], 1);
        int hash = hashFunction(item, bucket_size);
        while (true) {
            if (hash_table[hash + insert_start ] == -1) {
                hash_table[hash + insert_start] = item;
                break;
            }
            hash = (hash + 1) % bucket_size;
        }
    }
}
    
''', 'hash_table')


number_of_new_candidates_to_generate = cp.RawKernel(
    r"""

#define uint32_t unsigned int

extern "C" __global__
void number_of_new_candidates_to_generate(
    uint32_t *candidates, uint32_t numberOfKeys, 
    uint32_t keySize, uint32_t *numNewCands
    )
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numberOfKeys) return;
                            
    uint32_t i = tid;
    
    if (keySize == 1)
    {
        numNewCands[i + 1] = numberOfKeys - i - 1;
        return;
    }
    
    for (uint32_t j = i + 1; j < numberOfKeys; j++)
    {
        for (uint32_t k = 0; k < keySize; k++)
        {
            if (k == keySize - 1)
            {
                numNewCands[i + 1] += 1;                     
            }
            else
            {
                if (candidates[i * keySize + k] != candidates[j * keySize + k])
                {
                    return;
                }
            }
        }
    }
}
                            """,
    "number_of_new_candidates_to_generate",
)


write_the_new_candidates = cp.RawKernel(
    r"""

#define uint32_t unsigned int
extern "C" __global__
void write_the_new_candidates(
    uint32_t *candidates, uint32_t numCands, uint32_t candSize,
    uint32_t *newCandidatesIndex, uint32_t *newCandidates
)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > 0) return;
    int index = 0;
    for (int i = 0; i < numCands; i++)
    {
        if (newCandidatesIndex[i] == newCandidatesIndex[i + 1]) continue;

        int numNewCands = newCandidatesIndex[i + 1] - newCandidatesIndex[i];

        for (int j = 0; j < numNewCands; j++)
        {
            for (int k = 0; k < candSize; k++)
            {
                newCandidates[index++] = candidates[i * candSize + k];
            }
            newCandidates[index++] = candidates[(i + 2 + j) * candSize - 1];
        }
    }
    
}
                            
                            """,
    "write_the_new_candidates",
)


calc_support = cp.RawKernel(r'''
                          
__device__ unsigned int pcg_hash(int input)
{
    int state = input * 747796405u + 2891336453u;
    int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash function
__device__ unsigned int hashFunction(int key, int tableSize)
{
    return pcg_hash(key) % tableSize;
}
             
                            
__device__ bool query_item(const int *hash_table, int start_search, int end_search, int item)
{

    int tableSize = end_search - start_search;

    int hashIdx = hashFunction(item, tableSize);

    while (true)
    {
        if (hash_table[hashIdx + start_search] == -1)
        {
            return 0;
        }
        if (hash_table[hashIdx + start_search] == item)
        {
            return 1;
        }
        // Handle collisions (linear probing)
        hashIdx = (hashIdx + 1) % tableSize;
    }
}
                            
extern "C" __global__ void calc_support(const int *hash_table, const int *indices,
                                        const int numLines, const int *candidates,
                                        const int numCands, const int keySize, int *supports) 
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numLines) return;
    
    int start = indices[tid];
    int end = indices[tid + 1];
    
    for (int i = 0; i < numCands; i++) {
        int count = 0;
        for (int j = 0; j < keySize; j++) {
            int item = candidates[i * keySize + j];
            if (query_item(hash_table, start, end, item)) {
                count++;
            }
            else {
                break;
            }
        }
        if (count == keySize) {
            atomicAdd(&supports[i], 1);
        }
        
    }
}
                            
        ''', 'calc_support')


calc_support_shared = cp.RawKernel(r'''
                          
__device__ unsigned int pcg_hash(int input)
{
    int state = input * 747796405u + 2891336453u;
    int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash function
__device__ unsigned int hashFunction(int key, int tableSize)
{
    return pcg_hash(key) % tableSize;
}
             
                            
__device__ bool query_item(const int *hash_table, int start_search, int end_search, int item)
{

    int tableSize = end_search - start_search;

    int hashIdx = hashFunction(item, tableSize);

    while (true)
    {
        if (hash_table[hashIdx + start_search] == -1)
        {
            return 0;
        }
        if (hash_table[hashIdx + start_search] == item)
        {
            return 1;
        }
        // Handle collisions (linear probing)
        hashIdx = (hashIdx + 1) % tableSize;
    }
}

extern __shared__ int shared_hash_table[];
                            
extern "C" __global__ void calc_support_shared(const int *hash_table, const int *indices,
                                        const int numLines, const int *candidates,
                                        const int numCands, const int keySize, int *supports) 
{
    int block_id = blockIdx.x;
    if (block_id >= numLines) return;
    
    int start = indices[block_id];
    int end = indices[block_id + 1];
    
    int block_size = blockDim.x;
    
    int thread_id = threadIdx.x;
    for (int i = thread_id; i < end - start; i += block_size) {
        shared_hash_table[i] = hash_table[start + i];
    }
    __syncthreads();
    
    int shared_start = 0;
    int shared_end = end - start;
    
    for (int i = thread_id; i < numCands; i += block_size) {
        int count = 0;
        for (int j = 0; j < keySize; j++) {
            int item = candidates[i * keySize + j];
            if (query_item(shared_hash_table, shared_start, shared_end, item)) {
                count++;
            }
            else {
                break;
            }
        }
        if (count == keySize) {
            atomicAdd(&supports[i], 1);
        }
    }

    
    
}
                            
        ''', 'calc_support_shared')

class cuFPMiner_hash:
    def __init__(self, file, minSup, sep, file_type, allocator = 'device', shared=False):
        self.file = file
        self.minSup = minSup
        self.sep = sep
        self.file_type = file_type
        self.block_size = 512
        self.Patterns = []
        self.shared = shared
        
        cp.cuda.MemoryPool().free_all_blocks()
        
        if allocator == 'device':
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        elif allocator == 'pinned':
            # cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
            cp.cuda.set_allocator(cp.cuda.PinnedMemoryPool().malloc)
        elif allocator == 'managed':
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        else:
            raise ValueError("Invalid allocator type. Choose 'device', 'pinned', or 'managed'.")
        
        device = cp.cuda.Device()

        # Get the device properties
        device_properties = cp.cuda.runtime.getDeviceProperties(device.id)

        # Get the maximum shared memory per block
        self.max_shared_mem = device_properties['sharedMemPerBlock']
        
                
        # warm up the GPU
        with kvikio.CuFile(file, "r") as f:
            pass
    
        cp.cuda.Device().use()
        
        
    def read_csv(self):
        # read data
        file_size = os.path.getsize(self.file)
        
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
        
        # print(file_data[newline_indices[0]:newline_indices[1]])
        buffer = file_data[newline_indices[0]:newline_indices[1]]
        # convert to string
        buffer = cp.asnumpy(buffer)
        buffer = buffer.tobytes().decode('utf-8')
        
        # total number of items
        items_per_line = cp.zeros(number_of_transactions + 1, dtype=cp.int32)
        get_items_per_line((number_of_transactions//self.block_size + 1,), (self.block_size,), (file_data, 
                                                        newline_indices, number_of_transactions, items_per_line, ord(self.sep)))  
        cp.cuda.Device().synchronize()
        # add 1 to all elements
        items_per_line = items_per_line + 1
        items_per_line[0] = 0
        
        parsed_indices = cp.cumsum(items_per_line).astype(cp.int32)
        number_of_items = parsed_indices[-1].astype(cp.int32).get()
        
        raw_data = cp.zeros(number_of_items, dtype=cp.int32)
        convert_char_file_to_int_file((number_of_transactions//self.block_size + 1,), (self.block_size,), 
                                      (file_data, newline_indices, number_of_transactions, parsed_indices, raw_data, ord(self.sep)))
        
        self.number_of_transactions = number_of_transactions
        
        
        return raw_data, parsed_indices
        
    def read_parquet(self):
        # read data
        self.df = cudf.read_parquet(self.file)
        
        # all values to int
        self.df = self.df.astype('int32')
    
        # Get lengths (non-null count per row)
        lengths = self.df.notnull().sum(axis=1).to_cupy()
        
        # Add 0 to the start and the cumulative sum to get indices
        lengths = cp.concatenate([cp.array([0]), cp.cumsum(lengths)]).astype(cp.int32)
        
        # replace nulls with -1
        self.df = self.df.fillna(-1)
        
        
        # Flatten the dataframe and remove NaNs
        flattened = self.df.to_cupy().flatten().astype(cp.int32)
        flattened = flattened[flattened != -1]
        
        self.number_of_transactions = cp.int32(len(lengths) - 1)
        
        return flattened, lengths
    
    def read_file(self):
        
        if self.file_type == 'csv':
            csr_data, indices = self.read_csv()
        elif self.file_type == 'parquet':
            csr_data, indices = self.read_parquet()
        else:
            raise ValueError("Invalid file type. Choose 'csv' or 'parquet'.")
        
        max_item = cp.max(csr_data).get()
        supports = cp.zeros(max_item + 1, dtype=cp.int32)
        
        self.load_factor = 2
        db_size = indices[-1].get() * self.load_factor
        # self.hash_table = cp.zeros(db_size * self.load_factor, dtype=cp.int32)
        self.hash_table = cp.full(db_size, -1, dtype=cp.int32)
        
        hash_table((self.number_of_transactions//self.block_size + 1,), (self.block_size,), 
                          (csr_data, indices, self.number_of_transactions, self.hash_table, self.load_factor, supports))
        
        self.indices = indices * self.load_factor
        
        return supports
        
        
        
    def mine(self):
        start = time.time()
        supports = self.read_file()
        print("Time to read file:", time.time() - start)
        # find frequent items
        try:
            # Attempt to perform the operation with CuPy
            cands = cp.where(supports >= self.minSup)[0].astype(cp.int32)
        except Exception as e:
            # Fallback to NumPy if there's an error
            print(f"CuPy operation failed: {e}. Falling back to NumPy.")
            cands = supports.get()  # Transfer data to host as NumPy array
            cands = np.where(cands >= self.minSup)[0]
            cands = cp.array(cands, dtype=cp.int32)  # Convert back to CuPy array

                
        patterns = [[cands.get(), supports[cands].get()]]
        
        key_size = 1
        
        if self.shared:
            # get max diff
            max_diff = cp.max(cp.diff(self.indices)).get()
            shared_memory_usage = max_diff * cp.int32().nbytes
        
        if self.shared > self.max_shared_mem:
            raise ValueError("Shared memory usage exceeds the maximum shared memory per block.")
        
        
        while len(cands) > 1:
            num_new_cands = cp.zeros(len(cands) + 1, dtype=cp.int32)
            number_of_new_candidates_to_generate((len(cands)//self.block_size + 1,), (self.block_size,), 
                                                 (cands, len(cands), key_size, num_new_cands))
            
            new_cands_index = cp.cumsum(num_new_cands).astype(cp.int32)
            num_new_cands = new_cands_index[-1].get()
    
    
            new_cands = cp.zeros((num_new_cands, (key_size + 1)), dtype=cp.int32)
            write_the_new_candidates((num_new_cands//self.block_size + 1,), (self.block_size,), 
                                    (cands, len(cands), key_size, new_cands_index, new_cands))
            
            
            key_size += 1
            
            supports = cp.zeros(num_new_cands, dtype=cp.int32)
            
            if self.shared:
                if self.shared > self.max_shared_mem:
                    raise ValueError("Shared memory usage exceeds the maximum shared memory per block.")
                calc_support_shared((self.number_of_transactions + 1,), (self.block_size,),
                            (self.hash_table, self.indices, self.number_of_transactions, new_cands, num_new_cands, key_size, supports), 
                            shared_mem=shared_memory_usage)
            else:
                # calc_support((self.hash_table, self.indices, self.number_of_transactions, new_cands, num_new_cands, key_size, supports))
                calc_support((self.number_of_transactions//self.block_size + 1,), (self.block_size,), 
                            (self.hash_table, self.indices, self.number_of_transactions, new_cands, num_new_cands, key_size, supports))

            # locations = cp.where(supports >= self.minSup)[0].get()
            try:
                locations = cp.where(supports >= self.minSup)[0].astype(cp.int32)
            except cp.cuda.runtime.CUDADriverError as e:
                print(f"CuPy CUDA error: {e}. Resetting device...")
                cp.cuda.runtime.deviceReset()
                raise
            except Exception as e:
                print(f"General error: {e}. Attempting fallback...")
                try:
                    supports_np = supports.get()  # May still trigger the same issue
                    locations = np.where(supports_np >= self.minSup)[0]
                except Exception as np_e:
                    print(f"Fallback to NumPy failed: {np_e}")
                    raise

                        
            cands = new_cands[locations]
            supports = supports[locations]
            patterns.append([cands.get(), supports.get()])
            
            del new_cands
            del supports
            del locations
            
        for patterns, supports in patterns:
            for i in range(len(patterns)):
                self.Patterns.append((patterns[i], supports[i]))
                
                
        end = time.time()
    
        self.runtime = end - start
    
    def getRuntime(self):
        return self.runtime
    
    def getPatterns(self):
        return self.Patterns    
    
    def savePatterns(self, file):
        with open(file, 'w') as f:
            for pattern, support in self.Patterns:
                f.write(str(pattern) + " " + str(support) + "\n")
        
    def printResults(self):
        print("Runtime:", self.runtime)
        print("Number of frequent patterns:", len(self.Patterns))
            
        
if __name__ == "__main__":
    
    obj = cuFPMiner_hash("/home/tarun/cuPAMI/datasets/Transactional_kosarak.csv", 1400, '\t', 'csv', 'managed', True)
    obj.mine()
    obj.printResults()
    
    # obj = cuFPMiner_hash("/home/tarun/cuPAMI/datasets/Transactional_kosarak.csv", 2000, '\t', 'csv', 'managed', True)
    # obj.block_size = 1024
    # obj.mine()
    # obj.printResults()
    

    
    # obj = cuFPMiner_hash("/home/tarun/cuPAMI/datasets/Transactional_kosarak.csv", 4000, '\t', 'csv', 'device', True)
    # obj.mine()
    # obj.printResults()
    
    # obj = cuFPMiner_hash("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.csv", 20, '\t', 'csv', 'device', True)
    # obj.mine()
    # obj.printResults()
    
    # obj = cuFPMiner_hash("/home/tarun/cuPAMI/datasets/Transactional_BMS_POS.parquet", 100, '\t', 'parquet', 'device')
    # obj.mine()
    # obj.printResults()
    
    # obj = cuFPMiner_hash("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.parquet", 20, '\t', 'parquet', 'device', True)
    # obj.mine()
    # obj.printResults()