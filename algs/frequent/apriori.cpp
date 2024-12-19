#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <map>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <unistd.h>
#include <limits>
#include <sys/resource.h>


long print_memory_usage() {
    // Open the /proc/self/statm file
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) {
        std::cerr << "Error: Could not open /proc/self/statm" << std::endl;
        return -1; // Return -1 to indicate an error
    }

    long rss_pages = 0;
    statm.ignore(std::numeric_limits<std::streamsize>::max(), ' '); // Skip the first value
    statm >> rss_pages; // Read the RSS (Resident Set Size) in pages
    statm.close(); // Close the file

    if (rss_pages == 0) {
        std::cerr << "Error: Failed to read RSS from /proc/self/statm" << std::endl;
        return -1;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // Page size in KB
    long rss_kb = rss_pages * page_size_kb; // Calculate RSS in KB
    long rss_mb = rss_kb / 1024; // Convert KB to MB

    return rss_mb;
}

class Apriori
{
private:
    std::string file;
    int minSup;
    char sep;
    int numCores;
    std::unordered_map<std::string, std::unordered_set<int>> indices_dict;
    std::vector<std::pair<std::vector<std::string>, int>> Patterns;
    std::string output;
    double runtime;

    void read_csv()
    {
        // Step 1: Read all lines from the file sequentially
        std::ifstream infile(file);
        if (!infile.is_open())
        {
            std::cerr << "Error: Could not open file " << file << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(infile, line))
            {
                if (!line.empty())
                    lines.push_back(line);
            }
        }
        infile.close();

        // Step 2: Process lines in parallel
        // We'll use a vector of local maps, one per thread
        int n = (int)lines.size();
        int max_threads = omp_get_max_threads();
        std::vector<std::map<std::string, std::vector<int>>> local_dicts(max_threads);

#pragma omp parallel for schedule(dynamic)
        for (int line_number = 0; line_number < n; line_number++)
        {
            // Handle UTF-8 BOM on the first line only
            std::string curr_line = lines[line_number];
            if (line_number == 0 && !curr_line.empty() && curr_line[0] == '\xEF')
            {
                curr_line.erase(0, 3);
            }

            std::stringstream ss(curr_line);
            std::string item;

            // Each thread updates its own local dictionary
            int tid = omp_get_thread_num();
            auto &local_dict = local_dicts[tid];

            while (std::getline(ss, item, sep))
            {
                // Trim whitespace
                item.erase(item.find_last_not_of(" \t\n\r") + 1);
                item.erase(0, item.find_first_not_of(" \t\n\r"));

                if (!item.empty())
                {
                    local_dict[item].push_back(line_number);
                }
            }
        }

        // Step 3: Merge local_dicts into indices_dict
        for (auto &ldict : local_dicts)
        {
            for (auto &kv : ldict)
            {
                auto &global_vec = indices_dict[kv.first];
                global_vec.insert(kv.second.begin(), kv.second.end());
            }
        }
    }

    void unordered_set_intersection(const std::unordered_set<int> &a, const std::unordered_set<int> &b, std::unordered_set<int> &result)
    {
        // find the smaller set
        if (a.size() > b.size())
        {
            unordered_set_intersection(b, a, result);
            return;
        }

        for (const auto &x : a)
        {
            if (b.find(x) != b.end())
            {
                result.insert(x);
            }
        }
    }

    void parallel_mine_step(const std::vector<std::pair<std::vector<std::string>, std::unordered_set<int>>> &cands,
                            std::vector<std::pair<std::vector<std::string>, std::unordered_set<int>>> &local_nCands,
                            size_t start, size_t end)
    {
        for (size_t i = start; i < end; ++i)
        {
            const auto &cand_i = cands[i].first;
            for (size_t j = i + 1; j < cands.size(); ++j)
            {
                const auto &cand_j = cands[j].first;
                if (std::equal(cand_i.begin(), cand_i.end() - 1, cand_j.begin()) && cand_i.back() != cand_j.back())
                {
                    std::unordered_set<int> intersection;
                    unordered_set_intersection(cands[i].second, cands[j].second, intersection);
                    if (intersection.size() >= minSup)
                    {
                        std::vector<std::string> nCand = cand_i;
                        nCand.push_back(cand_j.back());
                        local_nCands.push_back({nCand, intersection});
                    }
                }
            }
        }
    }

public:
    Apriori(const std::string &file, int minSup, char sep, int numCores, std::string output = "")
        : file(file), minSup(minSup), sep(sep), numCores(numCores), runtime(0.0) {}

    void mine()
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Step 1: Read the CSV file
        read_csv();

        // print time tot read
        std::cout << "Time to read: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() << " seconds\n";

        // Step 2: Initialize candidates with single-item patterns
        std::vector<std::pair<std::vector<std::string>, std::unordered_set<int>>> cands;
        for (const auto &[key, indices] : indices_dict)
        {
            if (indices.size() >= minSup)
            {
                cands.push_back({{key}, indices});
                Patterns.push_back({{key}, static_cast<int>(indices.size())});
            }
        }

        // sort candidates in descending order of support
        std::sort(cands.begin(), cands.end(), [](const auto &a, const auto &b) {
            return a.second.size() > b.second.size();
        });

        // Step 3: Iteratively mine patterns
        while (!cands.empty())
        {
            std::vector<std::pair<std::vector<std::string>, std::unordered_set<int>>> nCands;

            const size_t num_threads = numCores;
            const size_t chunk_size = (cands.size() + num_threads - 1) / num_threads;

            std::vector<std::thread> threads;
            std::vector<std::vector<std::pair<std::vector<std::string>, std::unordered_set<int>>>> thread_nCands(num_threads);

            for (size_t t = 0; t < num_threads; ++t)
            {
                size_t start_idx = t * chunk_size;
                size_t end_idx = std::min(start_idx + chunk_size, cands.size());

                threads.emplace_back(&Apriori::parallel_mine_step, this,
                                     std::cref(cands), std::ref(thread_nCands[t]), start_idx, end_idx);
            }

            for (auto &thread : threads)
            {
                if (thread.joinable())
                    thread.join();
            }

            for (const auto &local_nCands : thread_nCands)
            {
                nCands.insert(nCands.end(), local_nCands.begin(), local_nCands.end());
            }

            for (const auto &cand : nCands)
            {
                Patterns.push_back({cand.first, static_cast<int>(cand.second.size())});
            }

            cands = std::move(nCands);
        }

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration<double>(end - start).count();
    }

    void save(const std::string &output)
    {
        std::ofstream outfile(output);
        if (!outfile.is_open())
        {
            std::cerr << "Error: Could not open file " << output << std::endl;
            exit(EXIT_FAILURE);
        }

        for (const auto &pattern : Patterns)
        {
            for (const auto &item : pattern.first)
            {
                outfile << item << " ";
            }
            outfile << ": " << pattern.second << std::endl;
        }

        outfile.close();
    }

    void printResults() const
    {
        std::cout << "Runtime: " << runtime << " seconds" << std::endl;
        std::cout << "Number of patterns: " << Patterns.size() << std::endl;
        std::cout << "Memory usage: " << print_memory_usage() << " MB" << std::endl;
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        size_t peakMemory = usage.ru_maxrss;
        std::cout << "Peak memory usage: " << peakMemory / 1024 << " MB\n";
    }

    double getRuntime() const { return runtime; }
};

char parse_separator(const std::string &arg)
{
    if (arg == "\\s")
        return ' '; // Space
    if (arg == "\\t")
        return '\t'; // Tab
    if (arg == "\\n")
        return '\n'; // Newline
    if (arg == "\\r")
        return '\r'; // Carriage return
    if (arg == "\\0")
        return '\0'; // Null character
    if (arg.size() == 1)
        return arg[0]; // Single character
    std::cerr << "Error: Unsupported separator \"" << arg << "\"" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <file> <minSup> <separator> <numCores> <output>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string file = argv[1];
    int minSup = std::stoi(argv[2]);
    char sep = parse_separator(argv[3]);
    int numCores = std::stoi(argv[4]);
    std::string output = argv[5];

    Apriori apriori(file, minSup, sep, numCores, output);
    apriori.mine();
    apriori.printResults();
    apriori.save(output);

    return 0;
}

// g++ -std=c++20 -O3 -pthread -fopenmp apriori.cpp -o apriori