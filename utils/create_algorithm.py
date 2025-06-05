#!/usr/bin/env python3
"""Create skeleton directories for a new algorithm.

The script optionally scaffolds C++ or CUDA implementations alongside a
Python wrapper so the profiling utilities work consistently.
"""
import os
import sys
import argparse

PY_TEMPLATE = '''from algs.base_algorithm import BaseAlgorithm

class {class_name}(BaseAlgorithm):
    def mine(self):
        """Run the algorithm."""
        data_path = 'datasets/{name}/{name}.txt'
        data = self.readFile(data_path)
        # TODO: implement algorithm logic using `data`
        pass
'''

CPP_WRAPPER_TEMPLATE = '''from algs.base_algorithm import BaseAlgorithm
import os
import subprocess
import time
import psutil

class {class_name}(BaseAlgorithm):
    def mine(self):
        start = time.time()
        src_dir = os.path.join(os.path.dirname(__file__), 'src')
        binary = os.path.join(src_dir, '{name}')
        # TODO: compile and run the C++ implementation
        subprocess.run([binary], check=False)
        self.runtime = time.time() - start
        proc = psutil.Process(os.getpid())
        self.memoryRSS = proc.memory_info().rss
        self.memoryUSS = proc.memory_full_info().uss
'''

CPP_SRC_TEMPLATE = '''#include <iostream>

int main() {
    // TODO: implement algorithm
    std::cout << "Placeholder for {name}" << std::endl;
    return 0;
}
'''

CUDA_WRAPPER_TEMPLATE = CPP_WRAPPER_TEMPLATE

CUDA_SRC_TEMPLATE = '''#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel() {}

int main() {
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    std::cout << "Placeholder for {name} CUDA" << std::endl;
    return 0;
}
'''

def main():
    parser = argparse.ArgumentParser(description="Create a new algorithm skeleton")
    parser.add_argument("name", help="Algorithm name")
    parser.add_argument("--lang", choices=["python", "cpp", "cuda"], default="python",
                        help="Implementation language (default: python)")
    args = parser.parse_args()

    name = args.name
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = {
        'alg': os.path.join(root, 'algs', name),
        'exp': os.path.join(root, 'experiments', name),
        'data': os.path.join(root, 'datasets', name),
        'test': os.path.join(root, 'tests', name),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)
        gitkeep = os.path.join(p, '.gitkeep')
        open(gitkeep, 'a').close()

    alg_file = os.path.join(paths['alg'], f"{name}.py")
    if not os.path.exists(alg_file):
        class_name = ''.join(word.capitalize() for word in name.split('_'))
        if args.lang == 'python':
            template = PY_TEMPLATE
        elif args.lang == 'cpp':
            template = CPP_WRAPPER_TEMPLATE
        else:
            template = CUDA_WRAPPER_TEMPLATE
        with open(alg_file, 'w') as f:
            f.write(template.format(class_name=class_name, name=name))

    if args.lang in {'cpp', 'cuda'}:
        src_dir = os.path.join(paths['alg'], 'src')
        os.makedirs(src_dir, exist_ok=True)
        src_ext = 'cpp' if args.lang == 'cpp' else 'cu'
        src_file = os.path.join(src_dir, f"{name}.{src_ext}")
        if not os.path.exists(src_file):
            with open(src_file, 'w') as f:
                src_template = CPP_SRC_TEMPLATE if args.lang == 'cpp' else CUDA_SRC_TEMPLATE
                f.write(src_template.format(name=name))

    print(f"Created skeleton for '{name}'")


if __name__ == '__main__':
    main()
