#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <sstream>
#include <iterator>
#include <tuple>
#include <omp.h>
#include <unistd.h>
#include <mutex>
#include <functional>
#include <limits>
#include <sys/resource.h>


long print_memory_usage()
{
    // Open the /proc/self/statm file
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open())
    {
        std::cerr << "Error: Could not open /proc/self/statm" << std::endl;
        return -1; // Return -1 to indicate an error
    }

    long rss_pages = 0;
    statm.ignore(std::numeric_limits<std::streamsize>::max(), ' '); // Skip the first value
    statm >> rss_pages;                                             // Read the RSS (Resident Set Size) in pages
    statm.close();                                                  // Close the file

    if (rss_pages == 0)
    {
        std::cerr << "Error: Failed to read RSS from /proc/self/statm" << std::endl;
        return -1;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // Page size in KB
    long rss_kb = rss_pages * page_size_kb;            // Calculate RSS in KB
    long rss_mb = rss_kb / 1024;                       // Convert KB to MB

    return rss_mb;
}

struct Node
{
    std::vector<std::string> item;
    int count;
    Node *parent;
    std::unordered_map<std::string, Node *> children;

    Node(std::vector<std::string> item, int count, Node *parent)
        : item(std::move(item)), count(count), parent(parent) {}

    Node *add_child(const std::string &item, int count = 1)
    {
        auto [it, inserted] = children.try_emplace(item, nullptr);

        if (inserted)
        {
            // If the child was newly added
            it->second = new Node({item}, count, this);
        }
        else
        {
            // If the child already exists, increment its count
            it->second->count += count;
        }

        return it->second;
    }

    std::pair<std::vector<std::string>, int> traverse()
    {
        std::vector<std::string> transaction;
        transaction.reserve(32);
        int node_count = count;
        Node *curr = parent;
        for (Node *curr = parent; curr && curr->parent; curr = curr->parent)
        {
            transaction.emplace_back(curr->item.front());
        }
        return {std::move(transaction), node_count};
    }

    ~Node()
    {
        for (auto &[_, child] : children)
        {
            delete child; // Recursively delete all child nodes
        }
    }
};

char parse_separator(const std::string &arg)
{
    if (arg == "\\s")
        return ' ';
    if (arg == "\\t")
        return '\t';
    if (arg == "\\n")
        return '\n';
    if (arg == "\\r")
        return '\r';
    if (arg == "\\0")
        return '\0';
    if (arg.size() == 1)
        return arg[0];
    std::cerr << "Error: Unsupported separator \"" << arg << "\"" << std::endl;
    exit(EXIT_FAILURE);
}

struct VectorHash
{
    std::size_t operator()(const std::vector<std::string> &v) const
    {
        std::hash<std::string> hasher;
        std::size_t seed = 0;
        for (const auto &s : v)
        {
            seed ^= hasher(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class FPGrowth
{
private:
    std::map<std::vector<std::string>, int> final_patterns;
    std::string input_file;
    char separator;
    int min_support;
    double runtime;
    int num_cores;

    std::tuple<std::unordered_map<std::vector<std::string>, int, VectorHash>, std::unordered_map<std::string, int>>
    create_itemsets()
    {
        std::unordered_map<std::string, int> item_count;

        std::ifstream file(input_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Input file not found.");
        }

        // Read all lines into a vector first (done sequentially)
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(file, line))
            {
                // remove carriage return
                if (!line.empty())
                {
                    lines.push_back(line);
                }
            }
        }
        file.close();

        // Prepare thread-local structures
        int max_threads = omp_get_max_threads();
        std::vector<std::unordered_map<std::string, int>> local_counts(max_threads);
        std::vector<
            std::vector<std::pair<std::vector<std::string>, int>>>
            local_database(max_threads);

#pragma omp parallel
        {

#pragma omp for
            for (int i = 0; i < (int)lines.size(); i++)
            {
                std::string line = lines[i];

                int tid = omp_get_thread_num();

                // remove carriage return
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }

                std::stringstream ss(line);
                std::string item;
                std::vector<std::string> transaction;
                while (std::getline(ss, item, separator))
                {
                    // Trim whitespace
                    item.erase(item.find_last_not_of(" \t\n\r") + 1);
                    item.erase(0, item.find_first_not_of(" \t\n\r"));
                    if (item.empty())
                    {
                        continue;
                    }
                    transaction.push_back(item);

                    // Update thread-local item counts
                    local_counts[tid][item]++;
                }
                local_database[tid].push_back({std::move(transaction), 1});
            }
        }

        // Merge thread-local item counts
        for (auto &local_map : local_counts)
        {
            for (auto &kv : local_map)
            {
                item_count[kv.first] += kv.second;
            }
        }

        // Filter and sort transactions based on min_support
        std::unordered_map<std::vector<std::string>, int, VectorHash> database;

        for (const auto &ldb : local_database)
        {
            for (const auto &[transaction, count] : ldb)
            {
                if (transaction.empty())
                    continue;

                std::vector<std::string> filtered_transaction;
                filtered_transaction.reserve(transaction.size());
                for (const auto &item : transaction)
                {
                    if (item_count[item] >= min_support)
                    {
                        filtered_transaction.push_back(item);
                    }
                }

                if (!filtered_transaction.empty())
                {
                    std::sort(filtered_transaction.begin(), filtered_transaction.end());
                    database[filtered_transaction] += count;
                }
            }
        }

        return std::make_tuple(std::move(database), std::move(item_count));
    }

    std::tuple<Node *, std::unordered_map<std::string, std::pair<std::unordered_set<Node *>, int>>> construct(
        std::unordered_map<std::vector<std::string>, int, VectorHash> &database,
        const std::unordered_map<std::string, int> &item_counts)
    {
        std::unordered_map<std::string, std::pair<std::unordered_set<Node *>, int>> item_node;
        Node *global_root = new Node({}, 0, nullptr);

        for (const auto &[transaction, count] : database)
        {
            if (transaction.empty())
                continue;

            Node *curr = global_root;
            for (const auto &item : transaction)
            {
                curr = curr->add_child(item, count);
                item_node[item].first.insert(curr);
                item_node[item].second += count;
            }
        }

        return std::make_tuple(std::move(global_root), std::move(item_node));
    }

    void sequential_mining(Node *root, std::unordered_map<std::string, std::pair<std::unordered_set<Node *>, int>> &item_nodes, std::vector<std::string> pattern)
    {
// Shared structure for final patterns
#pragma omp parallel
#pragma omp single nowait
        {
            for (const auto &[item, node_info] : item_nodes)
            {
                const auto &[nodes, count] = node_info;

                // Process only items meeting the minimum support threshold
                if (count >= min_support)
                {
#pragma omp task firstprivate(item, nodes, count, pattern)
                    {
                        std::vector<std::string> new_pattern = pattern;
                        new_pattern.push_back(item);

// Update the final_patterns in a thread-safe manner
#pragma omp critical
                        {
                            final_patterns[new_pattern] = count;
                        }

                        // Thread-local structures for transaction databases and item counts
                        std::unordered_map<std::vector<std::string>, int, VectorHash> new_database;
                        std::unordered_map<std::string, int> new_item_counts;

                        // Traverse nodes and build the new database
                        for (Node *node : nodes)
                        {
                            auto [transaction, node_count] = node->traverse();
                            new_database[transaction] += node_count;

                            for (const auto &item : transaction)
                            {
                                new_item_counts[item] += node_count;
                            }
                        }

                        // Construct the conditional FP-tree for the new database
                        auto [new_root, new_item_nodes] = construct(new_database, new_item_counts);

                        // Recursive call to process the next level of the tree
                        sequential_mining(new_root, new_item_nodes, new_pattern);

                        // Cleanup memory for the newly constructed root
                        delete new_root;
                    }
                }
            }
        }
    }

public:
    FPGrowth(const std::string &iFile, int minSup, char sep, int cores = 1)
        : input_file(iFile), separator(sep), min_support(minSup), runtime(0), num_cores(cores) {}

    void mine()
    {
        auto start = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_cores);

        // create_itemsets();
        auto [db, ic] = create_itemsets();

        // print time to read
        std::cout << "Time to read: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() << " seconds\n";

        auto [root, item_nodes] = construct(db, ic);
        std::cout << "Time to construct: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() << " seconds\n";

        sequential_mining(root, item_nodes, {});

        delete root;

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration<double>(end - start).count();
    }

    void save(const std::string &outFile)
    {
        std::ofstream file(outFile);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open output file.");
        }
        for (const auto &[pattern, count] : final_patterns)
        {
            for (size_t i = 0; i < pattern.size(); ++i)
            {
                file << pattern[i];
                if (i < pattern.size() - 1)
                {
                    file << separator;
                }
            }
            file << ":" << count << "\n";
        }
        file.close();
    }

    void printResults()
    {
        std::cout << "Runtime: " << runtime << " seconds\n";
        std::cout << "Number of patterns: " << final_patterns.size() << "\n";
        std::cout << "Memory usage: " << print_memory_usage() << " MB\n";
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        size_t peakMemory = usage.ru_maxrss;
        std::cout << "Peak memory usage: " << peakMemory / 1024 << " MB\n";
    }
};

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

    try
    {
        FPGrowth fpgrowth(file, minSup, sep, numCores);
        fpgrowth.mine();
        fpgrowth.printResults();
        fpgrowth.save(output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// g++ -std=c++20 -O3 -fopenmp fpgrowth.cpp -o fpgrowth
