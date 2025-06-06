#!/usr/bin/env python3
"""Utility to scaffold a new algorithm, dataset and test."""

import argparse
import os
from textwrap import dedent

ALG_TEMPLATE = """\
from .base_algorithm import BaseAlgorithm

class {class_name}(BaseAlgorithm):
    def __init__(self, data=None):
        super().__init__()
        self.data = data

    def run(self):
        # TODO: implement algorithm
        self._runtime = 0
        self._memory_rss = 0
        self._memory_uss = 0
        return []
"""

TEST_TEMPLATE = """\
from algorithms.{name} import {class_name}

def test_{name}(tmp_path):
    alg = {class_name}()
    alg.run()
    assert alg.getruntime() is not None
"""

DATA_TEMPLATE = "1 2 3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Algorithm name")
    args = parser.parse_args()

    name = args.name.lower()
    class_name = ''.join(part.capitalize() for part in name.split('_'))

    os.makedirs('algorithms', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('tests', exist_ok=True)

    alg_path = os.path.join('algorithms', f"{name}.py")
    test_path = os.path.join('tests', f"test_{name}.py")
    data_path = os.path.join('datasets', f"{name}.txt")

    with open(alg_path, 'w') as f:
        f.write(ALG_TEMPLATE.format(class_name=class_name))

    with open(test_path, 'w') as f:
        f.write(TEST_TEMPLATE.format(name=name, class_name=class_name))

    with open(data_path, 'w') as f:
        f.write(DATA_TEMPLATE)

    print(f"Created {alg_path}, {test_path}, {data_path}")


if __name__ == "__main__":
    main()
