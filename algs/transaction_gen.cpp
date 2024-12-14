#include <iostream>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <random>
#include <algorithm>
#include <limits>
#include <thread>
#include <mutex>

std::mutex fileMutex; // Mutex for writing to the file

void generateLineLengths(int n, int totalLines, std::vector<int>& lineLengths, int totalItems, std::mt19937& gen) {
    int remainingItems = totalItems;
    for (int i = 0; i < totalLines; ++i) {
        if (i == totalLines - 1) {
            lineLengths[i] = std::max(1, remainingItems);
        } else {
            int maxLength = std::min(remainingItems - (totalLines - i - 1), 2 * n);
            std::uniform_int_distribution<> dist(1, maxLength);
            lineLengths[i] = dist(gen);
            remainingItems -= lineLengths[i];
        }
    }
}

void writeLineToFile(std::ofstream& outFile, int length, int maxValue, std::mt19937& gen) {
    std::unordered_set<int> uniqueNumbers;
    std::uniform_int_distribution<> dist(0, maxValue);

    while (static_cast<int>(uniqueNumbers.size()) < length) {
        uniqueNumbers.insert(dist(gen));
    }

    std::string line;
    for (int num : uniqueNumbers) {
        line += std::to_string(num) + " ";
    }
    line += '\n';

    std::lock_guard<std::mutex> lock(fileMutex);
    outFile << line;
}

void generateLines(const std::vector<int>& lineLengths, int maxValue, const std::string& fileName, int numThreads) {
    std::ofstream outFile(fileName, std::ios::out);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << fileName << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::thread> threads;
    int totalLines = lineLengths.size();

    for (int i = 0; i < totalLines; ++i) {
        threads.emplace_back([&outFile, &lineLengths, maxValue, &gen, i]() {
            writeLineToFile(outFile, lineLengths[i], maxValue, gen);
        });

        if (threads.size() == numThreads || i == totalLines - 1) {
            for (std::thread& t : threads) {
                t.join();
            }
            threads.clear();
        }
    }

    outFile.close();
}

int main() {
    int n; // Average line length
    int totalLines; // Number of lines
    std::string fileName = "lines.txt"; // Default output file

    // Input validation for average line length
    do {
        std::cout << "Enter the average line length (n, must be > 0): ";
        std::cin >> n;
        if (std::cin.fail() || n <= 0) {
            std::cerr << "Invalid input. Please enter a positive integer." << std::endl;
            std::cin.clear(); // Clear error flags
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
        }
    } while (n <= 0);

    // Input validation for total lines
    do {
        std::cout << "Enter the number of lines (must be > 0): ";
        std::cin >> totalLines;
        if (std::cin.fail() || totalLines <= 0) {
            std::cerr << "Invalid input. Please enter a positive integer." << std::endl;
            std::cin.clear(); // Clear error flags
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
        }
    } while (totalLines <= 0);

    std::cin.ignore(); // Clear newline character after integer input
    std::cout << "Enter the output file name (default is lines.txt): ";
    std::getline(std::cin, fileName);
    if (fileName.empty()) {
        fileName = "lines.txt";
    }

    // Total number of items required in the file
    int totalItems = n * totalLines;
    std::vector<int> lineLengths(totalLines);

    std::random_device rd;
    std::mt19937 gen(rd());

    generateLineLengths(n, totalLines, lineLengths, totalItems, gen);

    int maxValue = 2 * n;
    // int numThreads = std::thread::hardware_concurrency(); // Use all available cores

    int numThreads = 4; // Use 4 threads

    generateLines(lineLengths, maxValue, fileName, numThreads);

    std::cout << "Output written to file '" << fileName << "'." << std::endl;

    return 0;
}


// g++ transaction_gen.cpp -03 -o transaction_gen