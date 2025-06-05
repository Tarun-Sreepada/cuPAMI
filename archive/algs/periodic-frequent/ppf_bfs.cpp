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
#include <omp.h>
#include <numeric>
#include <cmath>
#include <iterator>

class PPF_BFS {
public:
    PPF_BFS(const std::string& file, int minSup, int maxPer, double per_ratio,
            const std::string& sep, const std::string& output_file, int num_cores)
        : file_(file), minSup_(minSup), maxPer_(maxPer), per_ratio_(per_ratio),
          sep_(sep), output_file_(output_file), num_cores_(num_cores) {}

    void mine() {
        auto start = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_cores_);

        auto items = read_file();

        std::vector<std::pair<std::vector<int>, std::set<int>>> cands;
        // Process single items
        for (auto &kv : items) {
            int k = kv.first;
            auto &v = kv.second;
            if ((int)v.size() >= minSup_) {
                int perSup = getPerSup(v);
                double ratio = (double)perSup / (double)(v.size()+1);
                if (ratio >= per_ratio_) {
                    Patterns_[std::vector<int>{k}] = { (int)v.size(), ratio };
                }
                cands.push_back({ {k}, v });
            }
        }

        // Sort candidates by their pattern lexicographically
        // This ensures that candidates with common prefixes are grouped
        std::sort(cands.begin(), cands.end(),
                  [](auto &a, auto &b){return a.first < b.first;});

        // BFS approach: repeatedly generate nCands from cands
        while (!cands.empty()) {
            std::vector<std::pair<std::vector<int>, std::set<int>>> nCands;

            // We use a nested loop: for each i, we try to combine with j > i
            // Parallelize the inner loop
            // We'll store results in thread-local vectors and merge later
            #pragma omp parallel
            {
                std::vector<std::pair<std::vector<int>, std::set<int>>> local_nCands;
                std::vector<std::pair<std::vector<int>, std::pair<int,double>>> local_patterns;

                #pragma omp for nowait
                for (size_t i = 0; i < cands.size(); i++) {
                    for (size_t j = i+1; j < cands.size(); j++) {
                        const std::vector<int> &k1 = cands[i].first;
                        const std::vector<int> &k2 = cands[j].first;
                        // Check if k1[:-1] == k2[:-1] and last element differ
                        if (can_join(k1, k2)) {
                            std::vector<int> nCand = k1;
                            nCand.push_back(k2.back());

                            // intersection of tids
                            const std::set<int> &s1 = cands[i].second;
                            const std::set<int> &s2 = cands[j].second;
                            std::set<int> v;
                            std::set_intersection(s1.begin(), s1.end(),
                                                  s2.begin(), s2.end(),
                                                  std::inserter(v, v.begin()));
                            if ((int)v.size() >= minSup_) {
                                int perSup = getPerSup(v);
                                double ratio = (double)perSup / (double)(v.size() + 1);
                                local_nCands.push_back({ nCand, v });
                                if (ratio >= per_ratio_) {
                                    local_patterns.push_back({ nCand, { (int)v.size(), ratio } });
                                }
                            }
                        } else {
                            // Since candidates are sorted, if prefixes don't match here,
                            // no further matches for this i with larger j
                            // Just break
                            break;
                        }
                    }
                }

                // Merge local_nCands and local_patterns into global structures
                #pragma omp critical
                {
                    for (auto &cand : local_nCands) {
                        nCands.push_back(std::move(cand));
                    }
                    for (auto &pat : local_patterns) {
                        Patterns_[pat.first] = pat.second;
                    }
                }
            }

            // Sort new candidates before next iteration
            std::sort(nCands.begin(), nCands.end(),
                      [](auto &a, auto &b){return a.first < b.first;});
            cands = std::move(nCands);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "Number of patterns found: " << Patterns_.size() << std::endl;
        std::cout << "Time taken: " << diff.count() << " s" << std::endl;

        if (!output_file_.empty()) {
            std::ofstream ofs(output_file_);
            if (ofs.is_open()) {
                for (auto &pat : Patterns_) {
                    ofs << "[ ";
                    for (auto &item : pat.first) ofs << item << " ";
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
    // Patterns: key = pattern (vector<int>), value = (support, ratio)
    std::map<std::vector<int>, std::pair<int,double>> Patterns_;

    static bool can_join(const std::vector<int> &k1, const std::vector<int> &k2) {
        // Check if k1[:-1] == k2[:-1] and last elements differ
        if (k1.size() != k2.size()) return false;
        for (size_t idx = 0; idx+1 < k1.size(); idx++) {
            if (k1[idx] != k2[idx]) return false;
        }
        // last element must differ
        return k1.back() != k2.back();
    }

    std::map<int, std::set<int>> read_file() {
        std::ifstream ifs(file_);
        if (!ifs) {
            std::cerr << "Cannot open file: " << file_ << "\n";
            exit(1);
        }

        // Determine delimiter
        char delimiter = ' ';
        if (sep_ == "\\t") {
            delimiter = '\t';
        } else if (sep_ == "\\s") {
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

        std::map<int, std::vector<int>> items_map;
        max_tid_ = 0;
        // first integer in row is tid, rest are items
        for (auto &transaction : file_data) {
            if (transaction.empty()) continue;
            int tid = transaction[0];
            if (tid > max_tid_) max_tid_ = tid;
            for (size_t i = 1; i < transaction.size(); i++) {
                int item = transaction[i];
                items_map[item].push_back(tid);
            }
        }

        // Filter items by minSup
        std::map<int, std::set<int>> items;
        for (auto &kv : items_map) {
            if ((int)kv.second.size() >= minSup_) {
                std::set<int> s(kv.second.begin(), kv.second.end());
                items.insert({ kv.first, s });
            }
        }

        // Sort items by frequency descending
        std::vector<std::pair<int,std::set<int>>> sorted_items(items.begin(), items.end());
        std::sort(sorted_items.begin(), sorted_items.end(),
                  [](auto &a, auto &b) {
                      return a.second.size() > b.second.size();
                  });

        std::map<int, std::set<int>> sorted_map;
        for (auto &kv : sorted_items) {
            sorted_map.insert(kv);
        }

        return sorted_map;
    }

    int getPerSup(const std::set<int> &arr) const {
        // arr plus 0 and max_tid
        std::set<int> vec(arr);

        vec.insert(0);
        vec.insert(max_tid_);
        std::vector<int> vec2(vec.begin(), vec.end());
        std::sort(vec2.begin(), vec2.end());

        int count = 0;
        for (size_t i = 1; i < vec.size(); i++) {
            // int diff = vec[i] - vec[i-1];
            int diff = vec2[i] - vec2[i-1];
            if (diff <= maxPer_) {
                count++;
            }
        }
        return count;
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

    PPF_BFS ppf(input_file, minSup, maxPer, per_ratio, sep, output_file, num_cores);
    ppf.mine();

    return 0;
}

// g++ -O3 -std=c++17 -fopenmp -o ppf_bfs ppf_bfs.cpp