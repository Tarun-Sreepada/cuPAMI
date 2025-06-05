#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <iomanip>

// Constants
const size_t BUFFER_SIZE = 128 * 1024; // 128KB

// Function to get the size of the file
size_t getFileSize(const std::string& filename) {
    struct stat st;
    if(stat(filename.c_str(), &st) != 0) {
        std::cerr << "Cannot determine size of " << filename << std::endl;
        return 0;
    }
    return st.st_size;
}



// Function to read a chunk of the file and process it with buffered reading
std::vector<std::vector<int>> readChunkBuffered(const std::string& file, char delimiter, size_t start, size_t end) {
    std::vector<std::vector<int>> result;
    std::ifstream infile(file, std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error opening file in thread." << std::endl;
        return result;
    }

    infile.seekg(start);
    size_t currentPos = start;
    std::string buffer;
    std::string leftover;

    // Reuse a single temporary buffer
    std::vector<char> tempBuffer(BUFFER_SIZE);

    // Reusable objects for parsing
    std::istringstream ss; // Reusable string stream
    std::string token;
    std::string line;

    while (currentPos < end) {
        size_t bytesToRead = BUFFER_SIZE;
        if (currentPos + bytesToRead > end) {
            bytesToRead = end - currentPos;
        }

        // Read a block of data into the reusable buffer
        infile.read(tempBuffer.data(), bytesToRead);
        size_t bytesRead = infile.gcount();

        if (bytesRead == 0) {
            break; // EOF or no more data
        }

        currentPos += bytesRead;

        // Append the read data to the buffer
        buffer.append(tempBuffer.data(), bytesRead);

        // Prepend any leftover from the previous read
        if (!leftover.empty()) {
            buffer = leftover + buffer;
            leftover.clear();
        }

        // Process complete lines
        size_t pos = 0;
        while ((pos = buffer.find('\n')) != std::string::npos) {
            line = buffer.substr(0, pos);
            // Remove possible carriage return for Windows line endings
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            buffer.erase(0, pos + 1); // Remove processed line from buffer

            // Process the line using the reusable string stream
            ss.clear();
            ss.str(line);
            std::vector<int> row;
            while (std::getline(ss, token, delimiter)) {
                try {
                    row.emplace_back(std::stoi(token));
                } catch (const std::invalid_argument& e) {
                    // Handle non-integer tokens if necessary
                    row.emplace_back(0); // Example: push back 0 for invalid tokens
                }
            }
            result.emplace_back(std::move(row));
        }

        // Any remaining data in buffer is a partial line
        leftover = buffer;
        buffer.clear();
    }

    // After reading all blocks, process any remaining partial line
    if (!leftover.empty()) {
        line = leftover;
        // Remove possible carriage return for Windows line endings
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!line.empty()) { // Avoid processing empty lines
            ss.clear();
            ss.str(line);
            std::vector<int> row;
            while (std::getline(ss, token, delimiter)) {
                try {
                    row.emplace_back(std::stoi(token));
                } catch (const std::invalid_argument& e) {
                    // Handle non-integer tokens if necessary
                    row.emplace_back(0); // Example: push back 0 for invalid tokens
                }
            }
            result.emplace_back(std::move(row));
        }
    }

    infile.close();
    return result;
}



void readFileAndProcess(const std::string& file, char delimiter, int numCores) {
    // Start the timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Get the file size
    size_t fileSize = getFileSize(file);
    if (fileSize == 0) {
        std::cerr << "Empty file or cannot determine file size." << std::endl;
        return;
    }

    // Calculate chunk sizes
    size_t chunkSize = fileSize / numCores;
    std::vector<std::pair<size_t, size_t>> chunks(numCores, {0, 0});
    std::vector<std::vector<std::vector<int>>> allData(numCores);

    // Parallelize the chunk boundary determination
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < numCores; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numCores -1) ? fileSize : (i + 1) * chunkSize;

        std::ifstream infile(file, std::ios::in | std::ios::binary);
        if (!infile.is_open()) {
            std::cerr << "Error opening file in thread " << i << std::endl;
            chunks[i] = {fileSize, fileSize}; // Assign empty chunk
            continue;
        }

        std::string line;

        // Adjust start to the next newline if not the first chunk
        if(i != 0) {
            infile.seekg(start);
            std::getline(infile, line); // Discard partial line
            start = infile.tellg();
            if (start == static_cast<size_t>(-1)) start = fileSize;
        }

        // Adjust end to the end of the current line if not the last chunk
        if(i != numCores -1) {
            infile.seekg(end);
            std::getline(infile, line); // Read to the end of the current line
            end = infile.tellg();
            if (end == static_cast<size_t>(-1)) end = fileSize;
        }

        chunks[i] = {start, end};
        allData[i] = readChunkBuffered(file, delimiter, start, end);
        infile.close();
    }

    // Stop the timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    // Calculate throughput (MB/s)
    double fileSizeMB = static_cast<double>(fileSize) / (1024 * 1024);
    double throughput = fileSizeMB / elapsed.count();

    // calculate number of bytes used by the allData vector
    size_t allDataSize = 0;
    for (const auto& data : allData) {
        for (const auto& row : data) {
            allDataSize += row.size() * sizeof(int);
        }
    }
    std::cout << "Total memory used by allData: " << allDataSize << " bytes" << std::endl;


    // Get memory usage of the process
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    size_t peakMemory = usage.ru_maxrss; // in kilobytes


    // Print the last 10 lines
    std::cout << "\nLast 10 lines:" << std::endl;
    size_t totalLines = 0;
    for (const auto& data : allData) {
        totalLines += data.size();
    }


    // Get number of lines in the last chunk. if it is less than 10, print all lines else print the last 10 lines
    size_t lastChunkLines = allData[numCores - 1].size();
    size_t startLine = (lastChunkLines > 10) ? lastChunkLines - 10 : 0;
    for (size_t i = startLine; i < lastChunkLines; ++i) {
        for (int j = 0; j < allData[numCores - 1][i].size(); ++j) {
            std::cout << allData[numCores - 1][i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Print throughput
    std::cout << "\nTotal lines processed: " << totalLines << std::endl;

    uint64_t total_items = 0;
    for (const auto& data : allData) {
        for (const auto& row : data) {
            total_items += row.size();
        }
    }
    std::cout << "Total # of items: " << total_items << std::endl;

    std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "\nTotal throughput: " << std::fixed << std::setprecision(2) 
              << throughput << " MB/s" << std::endl;

    std::cout << "Peak memory (total process): " << peakMemory << " KB" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <file> <delimiter> <num_cores>" << std::endl;
        return 1;
    }

    std::string file = argv[1];
    char delimiter = argv[2][0];
    int numCores = std::stoi(argv[3]);

    // Validate number of cores
    if(numCores < 1) {
        std::cerr << "Number of cores must be at least 1." << std::endl;
        return 1;
    }

    readFileAndProcess(file, delimiter, numCores);
    return 0;
}

// Compilation command remains the same:
// g++ -std=c++17 -fopenmp -O3 -o parallel_read parallel_read.cpp
