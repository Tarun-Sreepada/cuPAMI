import cupy
import kvikio
import pandas as pd
import numpy as np
import os
import cupy as cp
import time
import psutil


count_number_of_lines = cp.RawKernel(r'''

extern "C" __global__ void count_number_of_lines(const unsigned char *data, const int size, const int newLine, int *numLines) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        if (data[i] == newLine) {
            atomicAdd(numLines, 1);
        }

    }
}

''', 'count_number_of_lines')

line_bounds_and_timestamp = cp.RawKernel(r'''
                                
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

    extern "C" __global__ void line_bounds_and_timestamp(const unsigned char *data, const int size, 
                                            const int newLine, int *indexes, int *timestamps, const int separator) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < size) {
            if (data[i] == newLine) {
                // Atomically increment the index count and get the current index
                int index = atomicAdd(indexes, 1);
                indexes[index + 1] = i + 1;

                // Locate the separator and extract the integer value
                int j = i + 1;
                int count = 0;
                while (j < size && data[j] != newLine) {
                    if (data[j] == separator) {
                        break;
                    }
                    j++;
                }

                // Make sure the timestamp array is large enough and null-terminate it
                char timestamp[32];  // Assuming the timestamp will not exceed 31 characters
                for (int k = i + 1; k < j && count < sizeof(timestamp) - 1; k++) {
                    timestamp[count++] = data[k];
                }
                timestamp[count] = '\0';

                // Convert the extracted substring to an integer using the custom function
                timestamps[index] = my_atoi(timestamp);
            }
        }
    }


    ''', 'line_bounds_and_timestamp')


get_items_per_line = cp.RawKernel(r'''
extern "C" __global__ void get_items_per_line(const unsigned char *data,
                                const int *indexes, const int numLines, int *items_per_line, const int seperator) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numLines) {
        for (int j = indexes[i]; j < indexes[i + 1]; j++) {
            if (data[j] == seperator) {
                atomicAdd(&items_per_line[i + 1], 1);
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


extern "C" __global__ void convert_char_file_to_int_file(const unsigned char *data, const int *indexes, const int numLines, const int *itemsPerLine, int *rawData, const int seperator) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numLines) return;
                            
    int start = indexes[tid];
    int end = indexes[tid + 1];
    char buffer[32];
    int bufferIndex = 0;

    int j = itemsPerLine[tid];

    while (data[start] != seperator) {
            start++;
        }
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

get_items_supports = cp.RawKernel(r'''

extern "C" __global__ void get_items_supports(const int *rawData, const int rawDataSize, int *supports) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > rawDataSize) return;
    atomicAdd(&supports[rawData[i]], 1);
}

''', 'get_items_supports')


create_bitset_for_items = cp.RawKernel(r'''

extern "C" __global__ void create_bitset_for_items(const int *rawData, const int *indexes, const int numLines, 
                                unsigned int *bitSets, const int bitsetSize, int *timestamps, int *bitIndexLoc, const int *valid, int *supports) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numLines) return;

    int start = indexes[i];
    int end = indexes[i + 1];
    int bitIndex = timestamps[i] % 32;
    int bitSetIndex = timestamps[i] / 32;


    for (int j = start; j < end; j++) {
        int item = rawData[j];

        if (!valid[item]) continue;

        int location = bitIndexLoc[item];
        atomicAdd(&supports[location], 1);
        if (bitIndex < 32) {
            atomicOr(&bitSets[(location * bitsetSize) + bitSetIndex], 1 << (31 - bitIndex));
        }
    }
}


''', 'create_bitset_for_items')

get_items_periodicity = cp.RawKernel(
    r"""
                
#define int32_t int
#define uint32_t unsigned int
extern "C" __global__
void get_items_periodicity(
    uint32_t *bitValues, uint32_t arraySize,
    uint32_t numberOfKeys,
    uint32_t *period, uint32_t maxPeriod, uint32_t maxTimeStamp
)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= numberOfKeys) return;

    uint32_t maxPeriodFound = 0;
    uint32_t lastSetBit = 0xFFFFFFFF; // Initialize to an invalid value
    uint32_t traversed = 0;

    for (uint32_t i = 0; i < arraySize; i++) {
        uint32_t intersection = bitValues[tid * arraySize + i];

        // Process 32 bits of the current word
        while (intersection) {
            uint32_t leadingZeros = __clz(intersection); // Count leading zeros
            uint32_t currentBit = traversed + 31 - leadingZeros;

            if (lastSetBit != 0xFFFFFFFF) { // If this isn't the first set bit
                uint32_t diff = currentBit - lastSetBit;
                if (diff > maxPeriodFound) {
                    maxPeriodFound = diff;
                }
            }

            lastSetBit = currentBit;

            // Clear the current bit
            intersection ^= (1 << (31 - leadingZeros));
        }

        traversed += 32;
        if (traversed > maxTimeStamp + 1) {
            break;
        }
    }

    // Handle the remaining period after the last set bit
    if (lastSetBit != 0xFFFFFFFF && (maxTimeStamp - lastSetBit) > maxPeriodFound) {
        maxPeriodFound = maxTimeStamp - lastSetBit;
    }

    period[tid] = maxPeriodFound;
}

""",
    "get_items_periodicity",
)


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


supportAndPeriod = cp.RawKernel(
    r"""
#define int32_t int
#define uint32_t unsigned int

extern "C" __global__
void supportAndPeriod(
    uint32_t *bitValues, uint32_t arraySize,
    uint32_t *candidates, uint32_t numberOfKeys, uint32_t keySize,
    uint32_t *support, uint32_t *period,
    uint32_t maxPeriod, uint32_t maxTimeStamp
)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numberOfKeys) return;

    uint32_t supportCount = 0;
    uint32_t periodCount = 0;
    uint32_t traversed = 0;
    uint32_t lastSetBit = 0xFFFFFFFF; // Initialize to an invalid value

    for (uint32_t i = 0; i < arraySize; i++) {
        uint32_t intersection = 0xFFFFFFFF;  // Start with all bits set to 1

        // Perform bitwise AND across all candidate bitValues
        for (uint32_t j = tid * keySize; j < (tid + 1) * keySize; j++) {
            intersection &= bitValues[candidates[j] * arraySize + i];
        }

        // Update support count using __popc
        supportCount += __popc(intersection);

        // Process the bits in the current word
        while (intersection) {
            uint32_t leadingZeros = __clz(intersection); // Find the first set bit
            uint32_t currentBit = traversed + 31 - leadingZeros;

            if (lastSetBit != 0xFFFFFFFF) { // Not the first set bit
                uint32_t diff = currentBit - lastSetBit;
                if (diff > period[tid]) {
                    period[tid] = diff;
                }
            }

            lastSetBit = currentBit;

            // Clear the current bit
            intersection ^= (1 << (31 - leadingZeros));
        }

        traversed += 32;

        // Check maxTimeStamp limit
        if (traversed > maxTimeStamp + 1) {
            break;
        }
    }

    // Handle the remaining period after the last set bit
    if (lastSetBit != 0xFFFFFFFF && (maxTimeStamp - lastSetBit) > period[tid]) {
        period[tid] = maxTimeStamp - lastSetBit;
    }

    // Store the results
    support[tid] = supportCount;
}

""",
    "supportAndPeriod",
)

class cuPFPMiner:
    def __init__(self, file, minSup, maxPer, sep, output_file, allocator='device'):
        self.file = file
        self.minSup = minSup
        self.maxPer = maxPer
        self.sep = sep
        self.output_file = kvikio.CuFile(output_file, "w")
        
        if allocator == 'device':
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        elif allocator == 'pinned':
            cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        elif allocator == 'managed':
            cp.cuda.set_allocator(cp.cuda.malloc_managed)
        else:
            raise ValueError("Invalid allocator type. Choose 'device', 'pinned', or 'managed'.")

        # warm up the GPU
        with kvikio.CuFile(file, "r") as f:
            pass
    
    def readFile(self, blockSize = 32):
        # Get file size to allocate memory in GPU
        file_size = os.path.getsize(self.file)

        # Allocate memory in GPU
        file_data = cp.empty(file_size * 8, dtype=cp.uint8)

        # Read file data
        with kvikio.CuFile(self.file , "r") as f:
            f.read(file_data)

        # Convert seperator and new line to bytes to find them in the file
        new_line = "\n"
        seperator_byte = ord(self.sep)
        new_line_byte = ord(new_line)

        # Allocate memory in GPU for number of lines and timestamps
        number_of_lines = cp.array([0], dtype=cp.int32)
        count_number_of_lines((file_size//blockSize + 1,), (blockSize,), (file_data, file_size, new_line_byte, number_of_lines)) 

        number_of_lines = number_of_lines[0].get()

        indexes = cp.zeros(number_of_lines + 1, dtype=cp.int32)
        timestamps = cp.zeros(number_of_lines, dtype=cp.int32)

        # Find line bounds and timestamps
        line_bounds_and_timestamp((file_size//blockSize + 1,), (blockSize,), (file_data, file_size, new_line_byte, indexes, timestamps, seperator_byte))

        # Sort indexes
        indexes[0] = 0  
        indexes = cp.sort(indexes)

        # get first timestamp because it is not calculated in the kernel function
        data = file_data[0:indexes[1]].get()
        buffer = []
        for i in range(data.size):
            buffer.append(int(data[i]))
            if data[i] == seperator_byte:
                buffer.pop()
                break
        
        timestamp = ""
        for i in range(len(buffer)):
            timestamp += chr(buffer[i])

        timestamps = cp.sort(timestamps).astype(cp.int32)
        timestamps[0] = int(timestamp)

        # Allocate memory for items per line to calculate the number of items in each line
        items_per_line = cp.zeros(number_of_lines + 1, dtype=cp.int32)
        get_items_per_line((file_size//blockSize + 1,), (blockSize,), (file_data, indexes, number_of_lines, items_per_line, seperator_byte))

        # Calculate the number of items in the file
        items_per_line = cp.cumsum(items_per_line).astype(cp.int32)
        number_of_items = items_per_line[number_of_lines].get()

        raw_data = cp.zeros(number_of_items, dtype=cp.int32)
        convert_char_file_to_int_file((file_size//blockSize + 1,), (blockSize,), (file_data, indexes, number_of_lines,items_per_line, raw_data, seperator_byte))

        del file_data
        del data
        del indexes
        del buffer
        del timestamp

        return raw_data, items_per_line, timestamps, number_of_lines, number_of_items

    def mine(self, blockSize = 32):
        start = time.time()
        raw_data, line_bounds, timestamps, number_of_lines, number_of_items = self.readFile(blockSize)
        print("Time to read file: ", time.time() - start)

        unique_items = cp.unique(raw_data)
        largest_item = int(cp.max(unique_items))

        item_supports = cp.zeros(largest_item + 2, dtype=cp.int32) # +2 to avoid out of bound error for conversion
        get_items_supports((number_of_items//blockSize + 1,), (blockSize,), (raw_data, number_of_items, item_supports))

        item_supports = cp.where(item_supports < self.minSup, 0, 1).astype(cp.int32)
        
        validty = item_supports

        rename_old_to_new = validty.copy()
        rename_old_to_new = cp.roll(rename_old_to_new, 1)
        rename_old_to_new = cp.cumsum(rename_old_to_new).astype(cp.int32)
        self.output_file.write(rename_old_to_new)

        integers_per_item_for_bitsets = number_of_lines // 32 + 1
        number_of_valid_items = int(rename_old_to_new[-1].get()) + 1

        supports = cp.zeros(number_of_valid_items, dtype=cp.int32)
        bitsets = cp.zeros((number_of_valid_items, integers_per_item_for_bitsets), dtype=cp.uint32)

        create_bitset_for_items((number_of_lines//blockSize + 1,), (blockSize,), 
        (raw_data, line_bounds, number_of_lines, bitsets, integers_per_item_for_bitsets, timestamps, rename_old_to_new, validty, supports))

        periodicity = cp.zeros(number_of_valid_items, dtype=cp.int32)
        max_timestamp = int(cp.max(timestamps))

        get_items_periodicity((largest_item//blockSize + 1,), (blockSize,), (bitsets, integers_per_item_for_bitsets, number_of_valid_items, periodicity, self.maxPer, max_timestamp))

        candidates = cp.where((supports >= self.minSup) & (periodicity <= self.maxPer))[0].astype(cp.uint32)
        key_size = 1

        patterns = len(candidates)

        while len(candidates) > 1:
            print("Number of Candidates: ", len(candidates))
            num_new_candidates = cp.zeros(len(candidates) + 1, dtype=cp.uint32)
            number_of_new_candidates_to_generate((len(candidates)//blockSize + 1,), (blockSize,), (candidates, len(candidates), key_size, num_new_candidates))

            new_candidates_index = cp.cumsum(num_new_candidates).astype(cp.uint32)
            num_new_candidates = new_candidates_index[-1].get()

            new_candidates = cp.zeros((num_new_candidates,(key_size + 1)), dtype=cp.uint32)
            write_the_new_candidates((num_new_candidates//blockSize + 1,), (blockSize,), (candidates, len(candidates), key_size, new_candidates_index, new_candidates))

            key_size += 1

            supports = cp.zeros(num_new_candidates, dtype=cp.int32)
            periodicity = cp.zeros(num_new_candidates, dtype=cp.int32)
            supportAndPeriod((num_new_candidates//blockSize + 1,), (blockSize,), 
                (bitsets, integers_per_item_for_bitsets, new_candidates, num_new_candidates, cp.int32(key_size),
                supports, periodicity, self.maxPer, max_timestamp))
            cp.cuda.device.Device().synchronize()

            locations = cp.where((supports >= self.minSup) & (periodicity <= self.maxPer))[0].astype(cp.uint32)

            self.output_file.write(num_new_candidates)
            self.output_file.write(cp.int32(key_size))
            self.output_file.write(candidates[locations])
            self.output_file.write(supports[locations])
            self.output_file.write(periodicity[locations])

            candidates = new_candidates[locations].astype(cp.uint32)


            patterns += len(candidates)


        self.number_of_patterns = patterns
        self.runtime = time.time() - start
        pid = os.getpid()
        pid = psutil.Process(pid)
        self.memoryRSS = pid.memory_info().rss
        self.memoryUSS = pid.memory_full_info().uss
        mempool = cp.get_default_memory_pool()

        # Get the total memory allocated by CuPy (in bytes)
        total_memory = mempool.total_bytes()

        # Get the memory currently in use (in bytes)
        used_memory = mempool.used_bytes()

        self.GPU_memory = used_memory

    def getRuntime(self):
        return self.runtime

    def getMemoryRSS(self):
        return self.memoryRSS
    
    def getMemoryUSS(self):
        return self.memroyUSS

    def getGPUMemory(self):
        return self.GPU_memory

    def printResults(self):
        print("Number of Patterns: ", self.number_of_patterns)
        print("Runtime: ", self.runtime)
        print("Memory RSS: ", self.memoryRSS)
        print("Memory USS: ", self.memoryUSS)
        print("GPU Memory: ", self.GPU_memory)


        


if __name__ == "__main__":
    file = "/home/tarun/cuPAMI/datasets/Temporal_kosarak.csv"
    minSup = 100
    maxPer = 100000
    sep = "\t"
    output_file = "output.txt"

    miner = cuPFPMiner(file, minSup, maxPer, sep, output_file, 'managed')
    miner.mine()
    miner.printResults()

