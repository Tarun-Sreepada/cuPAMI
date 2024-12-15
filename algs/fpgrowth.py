from typing import List, Dict, Tuple, Any, Union
from collections import Counter
from itertools import combinations
import pandas as pd
import time

class Node:
    def __init__(self, item: Any, count: int, parent: Union['Node', None]) -> None:
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

    def add_child(self, item: Any, count: int = 1) -> 'Node':
        if item not in self.children:
            self.children[item] = Node(item, count, self)
        else:
            self.children[item].count += count
        return self.children[item]

    def traverse(self) -> Tuple[List[Any], int]:
        transaction = []
        node = self.parent
        while node and node.parent:
            transaction.append(node.item)
            node = node.parent
        return transaction[::-1], self.count

class FPGrowth:
    def __init__(self, dataset: Union[str, pd.DataFrame], min_support: int, separator: str = '\t') -> None:
        self.min_support = min_support
        self.separator = separator
        self.dataset = dataset
        self.transactions = []
        self.patterns = {}
        self.runtime = 0.0

    def _load_data(self) -> None:
        """Load transactions from a file or DataFrame."""
        if isinstance(self.dataset, pd.DataFrame):
            if 'Transactions' not in self.dataset.columns:
                raise ValueError("DataFrame must contain a 'Transactions' column.")
            self.transactions = self.dataset['Transactions'].apply(lambda x: x.split(self.separator)).tolist()
        elif isinstance(self.dataset, str):
            with open(self.dataset, 'r', encoding='utf-8') as file:
                self.transactions = [line.strip().split(self.separator) for line in file if line.strip()]
        else:
            raise ValueError("Unsupported dataset type. Use a file path or a DataFrame.")

    def _build_tree(self, item_counts: Dict[Any, int]) -> Tuple[Node, Dict[Any, Tuple[set, int]]]:
        """Build the FP-tree and return the root and item nodes."""
        filtered_items = {k: v for k, v in item_counts.items() if v >= self.min_support}
        root = Node(None, 0, None)
        item_nodes = {}

        for transaction in self.transactions:
            current_node = root
            sorted_items = sorted(
                [item for item in transaction if item in filtered_items],
                key=lambda x: filtered_items[x],
                reverse=True
            )
            for item in sorted_items:
                current_node = current_node.add_child(item)
                if item not in item_nodes:
                    item_nodes[item] = (set(), 0)
                item_nodes[item][0].add(current_node)
                item_nodes[item] = (item_nodes[item][0], item_nodes[item][1] + 1)

        return root, item_nodes

    @staticmethod
    def _generate_combinations(items: List[Any]) -> List[Tuple[Any, ...]]:
        """Generate all non-empty combinations of a list."""
        return [comb for r in range(1, len(items) + 1) for comb in combinations(items, r)]

    def _mine_tree(self, node: Node, item_nodes: Dict[Any, Tuple[set, int]], patterns: Dict[Tuple[Any, ...], int]) -> None:
        """Recursive method to mine patterns from the FP-tree."""
        sorted_items = sorted(item_nodes.items(), key=lambda x: x[1][1])

        for item, (nodes, count) in sorted_items:
            if count < self.min_support:
                continue

            new_pattern = tuple(node.item + [item] if node.item else [item])
            patterns[new_pattern] = count

            conditional_item_counts = Counter()
            conditional_transactions = {}

            for n in nodes:
                transaction, trans_count = n.traverse()
                for trans_item in transaction:
                    conditional_item_counts[trans_item] += trans_count
                conditional_transactions[tuple(transaction)] = trans_count

            filtered_item_counts = {k: v for k, v in conditional_item_counts.items() if v >= self.min_support}
            if not filtered_item_counts:
                continue

            conditional_root = Node(None, 0, None)
            conditional_item_nodes = {}

            for transaction, trans_count in conditional_transactions.items():
                sorted_transaction = sorted(
                    [i for i in transaction if i in filtered_item_counts],
                    key=lambda x: filtered_item_counts[x],
                    reverse=True
                )
                current_node = conditional_root
                for i in sorted_transaction:
                    current_node = current_node.add_child(i, trans_count)
                    if i not in conditional_item_nodes:
                        conditional_item_nodes[i] = (set(), 0)
                    conditional_item_nodes[i][0].add(current_node)
                    conditional_item_nodes[i] = (conditional_item_nodes[i][0], conditional_item_nodes[i][1] + trans_count)

            if conditional_item_nodes:
                self._mine_tree(conditional_root, conditional_item_nodes, patterns)

    def mine(self) -> None:
        """Execute the FP-Growth algorithm to mine frequent patterns."""
        start_time = time.time()

        self._load_data()

        # Count item frequencies
        item_counts = Counter()
        for transaction in self.transactions:
            item_counts.update(transaction)

        # Build FP-tree
        root, item_nodes = self._build_tree(item_counts)

        # Mine patterns
        self._mine_tree(root, item_nodes, self.patterns)

        self.runtime = time.time() - start_time

    def save_patterns(self, output_file: str, separator: str = '\t') -> None:
        """Save mined patterns to a file."""
        with open(output_file, 'w') as f:
            for pattern, count in self.patterns.items():
                f.write(f"{separator.join(pattern)}:{count}\n")

    def get_patterns(self) -> Dict[Tuple[Any, ...], int]:
        """Return the mined patterns."""
        return self.patterns

    def get_runtime(self) -> float:
        """Return the runtime of the mining process."""
        return self.runtime

    def printResutls(self) -> None:
        """Print a summary of the mining results."""
        print(f"Runtime: {self.runtime:.2f} seconds")
        print(f"Total number of frequent patterns: {len(self.patterns)}")
        
        
if __name__ == "__main__":
    file = "/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.csv"
    sep = '\t'
    minSup = 20
    outFile = "patterns.txt"
    
    obj = FPGrowth(file, minSup, sep)
    obj.mine()
    obj.printResutls()