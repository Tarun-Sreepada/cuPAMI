#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <sstream>
#include <iterator>
#include <tuple>
#include <omp.h>
#include <unistd.h>

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
        if (children.find(item) == children.end())
        {
            children[item] = new Node({item}, count, this);
        }
        else
        {
            children[item]->count += count;
        }
        return children[item];
    }

    std::pair<std::vector<std::string>, int> traverse()
    {
        std::vector<std::string> transaction;
        int node_count = count;
        Node *curr = parent;
        while (curr && curr->parent)
        {
            transaction.push_back(curr->item.front());
            curr = curr->parent;
        }
        std::reverse(transaction.begin(), transaction.end());
        return {transaction, node_count};
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

class FPGrowth
{
private:
    std::vector<std::vector<std::string>> database;
    std::map<std::string, int> item_count;
    std::map<std::vector<std::string>, int> final_patterns;
    std::string input_file;
    char separator;
    int min_support;
    double runtime;
    int num_cores;

    void create_itemsets()
    {
        database.clear();
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
                line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
                if (!line.empty())
                {
                    lines.push_back(line);
                }
            }
        }
        file.close();

        // Prepare thread-local structures
        std::vector<std::vector<std::string>> local_transactions(lines.size());
        int max_threads = omp_get_max_threads();
        std::vector<std::map<std::string, int>> local_counts(max_threads);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)lines.size(); i++)
        {
            std::string line = lines[i];
            // Handle UTF-8 BOM
            if (!line.empty() && line[0] == '\xEF')
            {
                line.erase(0, 3);
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
                int tid = omp_get_thread_num();
                local_counts[tid][item]++;
            }
            local_transactions[i] = std::move(transaction);
        }

// Merge thread-local counts into global item_count
#pragma omp critical
        for (auto &local_map : local_counts)
        {
            for (auto &kv : local_map)
            {
                item_count[kv.first] += kv.second;
            }
        }

        // Move the transactions into the global database
        database = std::move(local_transactions);
    }

    std::tuple<Node *, std::map<std::string, std::pair<std::set<Node *>, int>>> construct(const std::map<std::string, int> &items)
    {
        std::map<std::string, std::pair<std::set<Node *>, int>> item_node;
        Node *root = new Node({}, 0, nullptr);
        for (const auto &transaction : database)
        {
            Node *curr_node = root;
            std::vector<std::string> sorted_items;
            for (const auto &it : transaction)
            {
                if (items.find(it) != items.end())
                {
                    sorted_items.push_back(it);
                }
            }

            // Sort the items in descending order of frequency
            std::sort(sorted_items.begin(), sorted_items.end(), [&](const std::string &a, const std::string &b)
                      { return items.at(a) > items.at(b); });

            for (const auto &it : sorted_items)
            {
                curr_node = curr_node->add_child(it);
                item_node[it].first.insert(curr_node);
                item_node[it].second++;
            }
            
        }
        return std::make_tuple(root, item_node);
    }

    void recursive_mining(Node *root, const std::map<std::string, std::pair<std::set<Node *>, int>> &item_node, int min_support)
    {

        // Collect items that meet the minimum support before parallelizing
        std::vector<std::pair<std::string, int>> freq_items;
        for (const auto &kv : item_node)
        {
            if (kv.second.second >= min_support)
            {
                freq_items.push_back({kv.first, kv.second.second});
            }
        }

        // Sort items by descending frequency
        std::sort(freq_items.begin(), freq_items.end(), [](const auto &a, const auto &b)
                  { return a.second > b.second; });

        std::map<std::vector<std::string>, int> local_patterns;

#pragma omp parallel
        {
            std::map<std::vector<std::string>, int> thread_local_patterns;

#pragma omp for nowait
            for (int i = 0; i < (int)freq_items.size(); i++)
            {
                const std::string &item = freq_items[i].first;
                const auto &node_info = item_node.at(item);

                // Extend pattern
                std::vector<std::string> new_itemset = root->item;
                new_itemset.push_back(item);
                thread_local_patterns[new_itemset] = node_info.second;

                // Build conditional pattern base
                std::map<std::string, int> item_count;
                std::vector<std::pair<std::vector<std::string>, int>> transactions;

                for (Node *node : node_info.first)
                {
                    auto [transaction, count] = node->traverse();
                    if (transaction.empty())
                        continue; // optionally skip empty
                    transactions.emplace_back(transaction, count);
                    for (const auto &trans_item : transaction)
                    {
                        item_count[trans_item] += count;
                    }
                }

                // Filter by min_support
                for (auto it = item_count.begin(); it != item_count.end();)
                {
                    if (it->second < min_support)
                        it = item_count.erase(it);
                    else
                        ++it;
                }

                if (item_count.empty())
                {
                    // No recursion needed if no conditional items remain
                    continue;
                }

                // Construct conditional FP-tree
                Node *new_root = new Node(new_itemset, 0, nullptr);
                std::map<std::string, std::pair<std::set<Node *>, int>> new_item_node;

                for (const auto &[transaction, count] : transactions)
                {
                    Node *curr_node = new_root;
                    std::vector<std::string> filtered_transaction;
                    for (const auto &it : transaction)
                    {
                        if (item_count.find(it) != item_count.end())
                            filtered_transaction.push_back(it);
                    }

                    // Sort by descending frequency
                    std::sort(filtered_transaction.begin(), filtered_transaction.end(), [&](const std::string &a, const std::string &b)
                              { return item_count[a] > item_count[b]; });

                    for (const auto &it : filtered_transaction)
                    {
                        curr_node = curr_node->add_child(it, count);
                        new_item_node[it].first.insert(curr_node);
                        new_item_node[it].second += count;
                    }
                }

                // Recursively mine conditional FP-tree if not empty
                if (!new_item_node.empty())
                {
                    // Use a non-parallel recursion for the conditional tree or replicate logic as needed
                    FPGrowth sub_fptree("", min_support, separator, num_cores);
                    sub_fptree.sequential_recursive_mining(new_root, new_item_node, min_support, thread_local_patterns);
                }
            }

#pragma omp critical
            {
                // Merge thread-local patterns into local_patterns
                for (const auto &kv : thread_local_patterns)
                {
                    local_patterns[kv.first] += kv.second;
                }
            }
        }

        // Merge local_patterns into final_patterns
        for (const auto &kv : local_patterns)
        {
            final_patterns[kv.first] = kv.second;
        }
    }

    // A helper function to allow recursion without re-using the outer class's parallelization setup.
    // This can be used for recursive calls within a parallel region.
    void sequential_recursive_mining(Node *root, const std::map<std::string, std::pair<std::set<Node *>, int>> &item_node,
                                     int min_support, std::map<std::vector<std::string>, int> &local_patterns)
    {
        for (const auto &[item, node_info] : item_node)
        {
            if (node_info.second < min_support)
            {
                continue;
            }

            std::vector<std::string> new_itemset = root->item;
            new_itemset.push_back(item);
            local_patterns[new_itemset] = node_info.second;

            std::map<std::string, int> item_count;
            std::vector<std::pair<std::vector<std::string>, int>> transactions;

            for (Node *node : node_info.first)
            {
                auto [transaction, count] = node->traverse();
                transactions.emplace_back(transaction, count);
                for (const auto &trans_item : transaction)
                {
                    item_count[trans_item] += count;
                }
            }

            for (auto it = item_count.begin(); it != item_count.end();)
            {
                if (it->second < min_support)
                {
                    it = item_count.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            if (item_count.empty())
            {
                continue;
            }

            Node *new_root = new Node(new_itemset, 0, nullptr);
            std::map<std::string, std::pair<std::set<Node *>, int>> new_item_node;

            for (const auto &[transaction, count] : transactions)
            {
                Node *curr_node = new_root;
                std::vector<std::string> filtered_transaction;
                for (const auto &item : transaction)
                {
                    if (item_count.find(item) != item_count.end())
                    {
                        filtered_transaction.push_back(item);
                    }
                }
                std::sort(filtered_transaction.begin(), filtered_transaction.end(), [&](const std::string &a, const std::string &b)
                          { return item_count[a] > item_count[b]; });

                for (const auto &it : filtered_transaction)
                {
                    curr_node = curr_node->add_child(it, count);
                    new_item_node[it].first.insert(curr_node);
                    new_item_node[it].second += count;
                }
            }

            if (!new_item_node.empty())
            {
                sequential_recursive_mining(new_root, new_item_node, min_support, local_patterns);
            }
        }
    }

public:
    FPGrowth(const std::string &iFile, int minSup, char sep, int cores = 1)
        : input_file(iFile), separator(sep), min_support(minSup), runtime(0), num_cores(cores) {}

    void mine()
    {
        auto start = std::chrono::high_resolution_clock::now();

        create_itemsets();

        // print time to read
        std::cout << "Time to read: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() << " seconds\n";

        for (auto it = item_count.begin(); it != item_count.end();)
        {
            if (it->second < min_support)
            {
                it = item_count.erase(it);
            }
            else
            {
                ++it;
            }
        }

        auto [root, item_node] = construct(item_count);

        // Set the number of threads
        omp_set_num_threads(num_cores);

        recursive_mining(root, item_node, min_support);

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
