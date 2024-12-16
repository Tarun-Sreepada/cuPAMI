import time
from typing import Any, Union, List, Tuple, Dict
from collections import Counter

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
    def __init__(self, file: str, minSup: int, sep: str):
        self.file = file
        self.minSup = minSup
        self.sep = sep
        self.transactions: List[Tuple[List[str], int]] = []
        self.patterns: Dict[Tuple[str, ...], int] = {}
        self.runtime = 0.0
        self.item_counts: Dict[str, int] = {}

    def read_file(self):
        # Read all lines and split by the given separator
        with open(self.file, 'r') as f:
            lines = f.readlines()

        lines = [line.strip().split(self.sep) for line in lines if line.strip()]
        self.item_counts = Counter()

        # Update global item counts
        for line in lines:
            self.item_counts.update(line)

        # Filter items by minSup
        for k in list(self.item_counts):
            if self.item_counts[k] < self.minSup:
                del self.item_counts[k]

        # Sort items in descending order by frequency
        self.item_counts = dict(sorted(self.item_counts.items(), key=lambda x: x[1], reverse=True))

        # Construct the filtered and sorted transactions
        for line in lines:
            filtered = [item for item in line if item in self.item_counts]
            if filtered:
                filtered.sort(key=lambda item: self.item_counts[item], reverse=True)
                self.transactions.append((filtered, 1))

    def _recursive_mine(self, root: Node, item_nodes: Dict[Any, Tuple[set, int]]) -> None:
        # Sort item_nodes by frequency (descending) and filter by minSup
        item_nodes = {k: v for k, v in sorted(item_nodes.items(), key=lambda x: x[1][1], reverse=True) if v[1] >= self.minSup}

        for item, (nodes, count) in item_nodes.items():
            # Create a new root that represents the current pattern
            new_pattern = root.item + [item] if isinstance(root.item, list) else [root.item, item]
            newRoot = Node(new_pattern, 0, None)

            # Record the pattern with its count
            self.patterns[tuple(newRoot.item)] = count

            newItemNodes: Dict[str, List[Any]] = {}
            itemCounts: Dict[str, int] = {}
            transactions: Dict[Tuple[str, ...], int] = {}

            # Build the conditional pattern base
            for node in nodes:
                transaction, cnt = node.traverse()
                if not transaction:
                    continue
                transaction_tuple = tuple(transaction)
                if transaction_tuple not in transactions:
                    transactions[transaction_tuple] = 0
                transactions[transaction_tuple] += cnt

                for it in transaction:
                    if it not in itemCounts:
                        itemCounts[it] = 0
                    itemCounts[it] += cnt

            # Filter items by minSup in the conditional pattern base
            itemCounts = {k: v for k, v in sorted(itemCounts.items(), key=lambda x: x[1], reverse=True) if v >= self.minSup}
            if not itemCounts:
                continue

            # Build the conditional FP-tree
            for transaction_tuple, cnt in transactions.items():
                # Filter the transaction by itemCounts and sort again
                newTransaction = [it for it in transaction_tuple if it in itemCounts]
                if not newTransaction:
                    continue
                newTransaction.sort(key=lambda it: itemCounts[it], reverse=True)

                curr = newRoot
                for it in newTransaction:
                    curr = curr.add_child(it, cnt)
                    if it not in newItemNodes:
                        newItemNodes[it] = [set(), 0]
                    newItemNodes[it][0].add(curr)
                    newItemNodes[it][1] += cnt

            # Recurse on the new conditional FP-tree
            if newItemNodes:
                self._recursive_mine(newRoot, newItemNodes)

    def mine(self) -> None:
        """Execute the FP-Growth algorithm to mine frequent patterns."""
        start_time = time.time()

        self.read_file()

        # Build the initial FP-tree
        root = Node([], 0, None)
        item_nodes: Dict[str, List[Any]] = {}

        for transaction, count in self.transactions:
            node = root
            for item in transaction:
                node = node.add_child(item, count)
                if item not in item_nodes:
                    item_nodes[item] = [set(), 0]
                item_nodes[item][0].add(node)
                item_nodes[item][1] += count

        # Mine the FP-tree recursively
        self._recursive_mine(root, item_nodes)

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

    def printResults(self) -> None:
        """Print a summary of the mining results."""
        print(f"Runtime: {self.runtime:.2f} seconds")
        print(f"Total number of frequent patterns: {len(self.patterns)}")


if __name__ == '__main__':
    file = "/home/tarun/cuPAMI/datasets/Transactional_retail.csv"
    sep = '\t'
    minSup = 50

    fpg = FPGrowth(file, minSup, sep)
    fpg.mine()
    fpg.printResults()