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
    if (tid != 0) start++;
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

convert_to_bitset = cp.RawKernel(r'''
extern "C" 
__global__ void convert_to_bitset(const int *data, const int *indexes, int numLines, int *bitset, int bitset_size, int *supports)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numLines) return;
    
    int bit_index = tid / 32;
    int offset = tid % 32;
    
    for (int i = indexes[tid]; i < indexes[tid + 1]; i++) {
        int item = data[i];
        atomicAdd(&supports[item], 1);
        atomicOr(&bitset[item * bitset_size + bit_index], 1 << offset);
    }
}
    
''', 'convert_to_bitset')

    
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
    uint32_t totalThreads = gridDim.x * blockDim.x;

    for (uint32_t i = tid; i < numCands; i += totalThreads)
    {
        if (newCandidatesIndex[i] == newCandidatesIndex[i + 1]) continue;
        int index = newCandidatesIndex[i] * (candSize + 1);
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
                                
#define uint32_t unsigned int
                                
extern "C" __global__
void calc_support(
    const int *bitValues, const int arraySize,
    const int *candidates, const int numberOfKeys, const int keySize,
    int *support)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfKeys) return;

    unsigned int supportCount = 0;

    // Loop over the bitValues array
    for (unsigned int i = 0; i < arraySize; ++i)
    {
        unsigned int intersection = 0xFFFFFFFF;
        for (unsigned int k = 0; k < keySize; ++k)
        {
            unsigned int item = candidates[tid * keySize + k];
            intersection &= bitValues[item * arraySize + i];
        }

        supportCount += __popc(intersection);
        
    }

    support[tid] = supportCount;
}


''', 'calc_support')

class cuFPMiner_bit:
    def __init__(self, file, minSup, sep, file_type, allocator = 'device'):
        self.file = file
        self.minSup = minSup
        self.sep = sep
        self.file_type = file_type
        self.block_size = 512
        self.Patterns = []
        
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
        
    def read_data(self):
        # read data
        if self.file_type == 'csv':
            csr_data, indices = self.read_csv()
        elif self.file_type == 'parquet':
            csr_data, indices = self.read_parquet()
        else:
            raise Exception("Invalid file type")
        
        max_item = cp.max(csr_data).get()
        # create biset
        self.bit_set_size = self.number_of_transactions // 32 + 1
        self.bit_set = cp.zeros((max_item + 1, self.bit_set_size), dtype=cp.int32)
        
        supports = cp.zeros(max_item + 1, dtype=cp.int32)   
        
        convert_to_bitset((self.number_of_transactions//self.block_size + 1,), (self.block_size,), 
                          (csr_data, indices, self.number_of_transactions, self.bit_set, self.bit_set_size, supports))
        
        
        return supports
        
        

    def mine(self):
        start = time.time()
        supports = self.read_data()
        self.time_to_read = time.time() - start
        
        # find frequent items
        cands = cp.where(supports >= self.minSup)[0].astype(cp.int32)
        
        patterns = [[cands.get(), supports[cands].get()]]
        
        key_size = 1
        
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
            
            calc_support((num_new_cands//self.block_size + 1,), (self.block_size,), 
                        (self.bit_set, self.bit_set_size, new_cands, num_new_cands, key_size, supports))
        
            
            locations = cp.where(supports >= self.minSup)[0].get()
            cands = new_cands[locations]
            supports = supports[locations]
            
            patterns.append([cands.get(), supports.get()])
            
        for patterns, supports in patterns:
            for i in range(len(patterns)):
                self.Patterns.append((patterns[i], supports[i]))
                
        # print("Number of frequent patterns:", len(self.Patterns))       
        end = time.time()
        self.runtime = end - start
    
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        used_mem = total_mem - free_mem
        self.memory = used_mem / 1024 / 1024
    
    def getRuntime(self):
        return self.runtime
    
    def getPatterns(self):
        return self.Patterns    

    def getMemoryRSS(self):
        return self.memory
        
    def getTimeToRead(self):
        return self.time_to_read
        
    def printResults(self):
        print("Runtime:", self.runtime)
        print("Number of frequent patterns:", len(self.Patterns))
        
    def savePatterns(self, file):
        with open(file, 'w') as f:
            for pattern, support in self.Patterns:
                f.write(str(pattern) + " " + str(support) + "\n")
        
        
        
if __name__ == "__main__":
    
    obj = cuFPMiner_bit("/home/tarun/cuPAMI/datasets/transactional/Transactional_kosarak.csv", 2000, '\t', 'csv', 'device')
    obj.mine()
    obj.printResults()
    
    # from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
    # fp = FPGrowth("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.csv", 20, '\t')
    # fp.mine()
    # fp.printResults()