#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <sys/resource.h>
#include <sys/time.h>

void readFileAndProcess(const std::string& file, char delimiter, int numCores) {
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Vector to hold the processed data
    std::vector<std::vector<int>> data;

    // Read the file
    std::ifstream infile(file);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << file << std::endl;
        return;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    infile.close();

    // Set the number of threads for OpenMP
    omp_set_num_threads(numCores);

    // Process lines in parallel
    data.resize(lines.size());
    #pragma omp parallel for
    for (size_t i = 0; i < lines.size(); ++i) {
        std::istringstream ss(lines[i]);
        std::string token;
        std::vector<int> row;
        while (std::getline(ss, token, delimiter)) {
            row.push_back(std::stoi(token));
        }
        data[i] = row;
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate memory usage
    size_t memoryUsed = 0;
    for (const auto& row : data) {
        memoryUsed += sizeof(row) + (row.size() * sizeof(int));
    }
    memoryUsed += sizeof(data);

    // Get memory usage of the process
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    size_t peakMemory = usage.ru_maxrss;

    // Print results
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Memory used (in program): " << memoryUsed / 1024 << " KB" << std::endl;
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

    readFileAndProcess(file, delimiter, numCores);
    return 0;
}

// g++ -std=c++17 -fopenmp -O3 -o parallel_read parallel_read.cpp