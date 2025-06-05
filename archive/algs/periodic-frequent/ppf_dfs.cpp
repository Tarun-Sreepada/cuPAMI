#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif

class PPF_DFS {
public:
    PPF_DFS(const std::string& file, int minSup, int maxPer, double per_ratio, 
            const std::string& sep, const std::string& output_file, int num_cores)
        : file_(file), minSup_(minSup), maxPer_(maxPer), per_ratio_(per_ratio), 
          sep_(sep), output_file_(output_file), num_cores_(num_cores) {}

    void mine() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Set number of threads
        #ifdef _OPENMP
        omp_set_num_threads(num_cores_);
        #endif

        // Read file and initialize items
        auto items = read_file();

        // Convert items to candidate format
        std::vector<std::pair<std::vector<int>, std::set<int>>> cands;
        for (auto& kv : items) {
            const int k = kv.first;
            auto& v = kv.second;
            int size_v = (int)v.size();
            if (size_v >= minSup_) {
                int perSup = getPerSup(v, max_tid_, maxPer_);
                double ratio = (double)perSup / (double)(size_v + 1);
                if (ratio >= per_ratio_) {
                    Patterns_[std::vector<int>{k}] = {size_v, ratio};
                }
                cands.push_back({{k}, v});
            }
        }

        recursive_single(cands);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "Number of patterns found: " << Patterns_.size() << std::endl;
        std::cout << "Time taken: " << diff.count() << " s" << std::endl;

        // Optionally write patterns to output_file_
        if (!output_file_.empty()) {
            std::ofstream ofs(output_file_);
            if (ofs.is_open()) {
                for (auto& pat : Patterns_) {
                    ofs << "[ ";
                    for (auto& item : pat.first) ofs << item << " ";
                    ofs << "] " << pat.second.first << " " << pat.second.second << "\n";
                }
            }
        }
    }

private:
    std::string file_;
    int minSup_;
    int maxPer_;
    double per_ratio_;
    std::string sep_;
    std::string output_file_;
    int num_cores_;

    int max_tid_ = 0;
    // Patterns: key = sorted vector of items, value = { support, ratio }
    std::map<std::vector<int>, std::pair<int,double>> Patterns_;
    std::mutex patterns_mutex_;

    std::map<int, std::set<int>> read_file() {
        std::ifstream ifs(file_);
        if (!ifs) {
            std::cerr << "Cannot open file: " << file_ << std::endl;
            exit(1);
        }

        // Determine delimiter
        // \s => space, \t => tab
        char delimiter = ' ';
        if (sep_ == "\\t") {
            delimiter = '\t';
        } else if (sep_ == "\\s") {
            // space delimiter is default above
            delimiter = ' ';
        }

        std::string line;
        std::vector<std::vector<int>> file_data;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;

            std::vector<int> row;
            if (delimiter == ' ') {
                // split by whitespace
                std::stringstream ss(line);
                int val;
                while (ss >> val) {
                    row.push_back(val);
                }
            } else {
                // split by tab
                size_t start = 0;
                while (true) {
                    size_t pos = line.find(delimiter, start);
                    if (pos == std::string::npos) {
                        // last token
                        if (start < line.size()) {
                            int val = std::stoi(line.substr(start));
                            row.push_back(val);
                        }
                        break;
                    } else {
                        if (pos > start) {
                            int val = std::stoi(line.substr(start, pos - start));
                            row.push_back(val);
                        }
                        start = pos + 1;
                    }
                }
            }
            if (!row.empty()) {
                file_data.push_back(row);
            }
        }

        // items: map from item -> set of TIDs
        std::map<int, std::vector<int>> items_map;
        max_tid_ = 0;
        for (auto &transaction : file_data) {
            if (transaction.empty()) continue;
            int tid = transaction[0];
            if (tid > max_tid_) max_tid_ = tid;
            for (size_t i = 1; i < transaction.size(); ++i) {
                int item = transaction[i];
                items_map[item].push_back(tid);
            }
        }

        // Keep only items with frequency >= minSup_
        std::map<int, std::set<int>> items;
        for (auto &kv : items_map) {
            if ((int)kv.second.size() >= minSup_) {
                std::set<int> s(kv.second.begin(), kv.second.end());
                items.insert({kv.first, s});
            }
        }

        // Sort items by frequency descending
        std::vector<std::pair<int,std::set<int>>> sorted_items(items.begin(), items.end());
        std::sort(sorted_items.begin(), sorted_items.end(), 
                  [](auto &a, auto &b) {
                      return a.second.size() > b.second.size();
                  });
        
        // Return as a map in that order
        std::map<int, std::set<int>> sorted_map;
        for (auto &kv : sorted_items) {
            sorted_map.insert(kv);
        }

        return sorted_map;
    }

    static int getPerSup(const std::set<int> &arr, int max_tid, int maxPer) {
        // copy set
        std::set<int> vec(arr);
        vec.insert(0);
        vec.insert(max_tid);
        // sort
        std::vector<int> sorted(vec.begin(), vec.end());
        int count = 0;
        for (size_t i = 1; i < sorted.size(); i++) {
            int diff = sorted[i] - sorted[i-1];
            if (diff <= maxPer) {
                count++;
            }
        }
        return count;
        
    }

    void add_pattern(const std::vector<int>& pattern, int support, double ratio) {
        std::lock_guard<std::mutex> lock(patterns_mutex_);
        Patterns_[pattern] = {support, ratio};
    }

    void recursive_single(const std::vector<std::pair<std::vector<int>, std::set<int>>> &cands) {
        for (size_t i = 0; i < cands.size(); i++) {
            std::vector<std::pair<std::vector<int>, std::set<int>>> newCands;
            
            // We'll parallelize this loop
            // We'll store thread-local patterns and then merge
            std::vector<std::pair<std::vector<int>, std::pair<int,double>>> local_patterns;
            #pragma omp parallel
            {
                std::vector<std::pair<std::vector<int>, std::pair<int,double>>> thread_local_patterns;
                std::vector<std::pair<std::vector<int>, std::set<int>>> thread_local_newCands;

                #pragma omp for nowait
                for (size_t j = i+1; j < cands.size(); j++) {
                    std::set<int> intersection_set;
                    const std::set<int> &s1 = cands[i].second;
                    const std::set<int> &s2 = cands[j].second;
                    // Set intersection
                    std::set_intersection(s1.begin(), s1.end(),
                                          s2.begin(), s2.end(),
                                          std::inserter(intersection_set, intersection_set.begin()));

                    int inter_size = (int)intersection_set.size();
                    if (inter_size >= minSup_) {
                        int perSup = getPerSup(intersection_set, max_tid_, maxPer_);
                        double ratio = (double)perSup / (double)(inter_size + 1);

                        std::vector<int> nCand = cands[i].first;
                        nCand.insert(nCand.end(), cands[j].first.begin(), cands[j].first.end());
                        std::sort(nCand.begin(), nCand.end());
                        nCand.erase(std::unique(nCand.begin(), nCand.end()), nCand.end());

                        thread_local_newCands.push_back({nCand, intersection_set});
                        if (ratio >= per_ratio_) {
                            thread_local_patterns.push_back({nCand, {inter_size, ratio}});
                        }
                    }
                }

                // Merge local results to the outer vectors
                #pragma omp critical
                {
                    for (auto &lp : thread_local_patterns) {
                        local_patterns.push_back(lp);
                    }
                    for (auto &nc : thread_local_newCands) {
                        newCands.push_back(nc);
                    }
                }
            }

            // Add collected patterns to global Patterns_
            for (auto &lp : local_patterns) {
                add_pattern(lp.first, lp.second.first, lp.second.second);
            }

            if (!newCands.empty()) {
                recursive_single(newCands);
            }
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <minSup> <maxPer> <per_ratio> <sep> <output_file> <num_cores>\n";
        return 1;
    }
    std::string input_file = argv[1];
    int minSup = std::stoi(argv[2]);
    int maxPer = std::stoi(argv[3]);
    double per_ratio = std::stod(argv[4]);
    std::string sep = argv[5];
    std::string output_file = argv[6];
    int num_cores = std::stoi(argv[7]);

    PPF_DFS ppf(input_file, minSup, maxPer, per_ratio, sep, output_file, num_cores);
    ppf.mine();

    return 0;
}

// g++ -O3 -std=c++17 -fopenmp -o ppf_dfs ppf_dfs.cpp
// ./ppf_dfs input.txt 2 10 0.5 \\s output.txt 4