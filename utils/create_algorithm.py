#!/usr/bin/env python3
"""Create skeleton directories for a new algorithm."""
import os
import sys

TEMPLATE = '''from algs.base_algorithm import BaseAlgorithm

class {class_name}(BaseAlgorithm):
    def mine(self):
        """Run the algorithm."""
        # TODO: implement algorithm logic
        pass
'''


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/create_algorithm.py <name>")
        sys.exit(1)

    name = sys.argv[1]
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
        with open(alg_file, 'w') as f:
            f.write(TEMPLATE.format(class_name=class_name))

    print(f"Created skeleton for '{name}'")


if __name__ == '__main__':
    main()
