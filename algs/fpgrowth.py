from typing import List, Dict, Tuple, Any, Union
from collections import Counter
from itertools import combinations
import pandas as pd
import time
from multiprocessing import Pool, Manager
import os


class Node:
    """
    Methods:
        __init__(item: Any, count: int, parent: Union['Node', None]) -> None:
            Initializes a Node with the given item, count, and parent.
    A class representing a node in an FP-growth tree.

    Attributes:
        item (Any): The item contained in this node.
        count (int): The count of the item.
        parent (Union['Node', None]): The parent node of this node.
        children (Dict[Any, 'Node']): A dictionary of child nodes.

    Methods:
        add_child(item: Any, count: int = 1) -> 'Node':
            Adds a child node with the given item and count. If the child node
            already exists, increments its count.

        traverse() -> Tuple[List[Any], int]:
            Traverses up the tree from this node to the root, collecting the
            items along the path and returning them as a transaction along with
            the count of this node.
    """
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
    """
    A class to represent the FP-Growth algorithm for mining frequent patterns from a dataset.
    Attributes:
    -----------
    file : str
        The path to the input file containing transactions.
    minSup : int
        The minimum support threshold for frequent patterns.
    sep : str
        The separator used in the input file to separate items in a transaction.
    transactions : list
        A list to store transactions after reading from the file.
    patterns : dict
        A dictionary to store the mined frequent patterns.
    runtime : float
        The time taken to execute the mining process.
    Methods:
    --------
    read_file():
        Reads the input file and processes transactions.
    _recursive_mine(root: 'Node', item_nodes: Dict[Any, Tuple[set, int]]) -> None:
        Recursively mines the FP-tree to find frequent patterns.
    mine() -> None:
        Executes the FP-Growth algorithm to mine frequent patterns.
    save_patterns(output_file: str, separator: str = '\t') -> None:
        Saves the mined patterns to a file.
    get_patterns() -> Dict[Tuple[Any, ...], int]:
        Returns the mined patterns.
    get_runtime() -> float:
        Returns the runtime of the mining process.
    printResults() -> None:
        Prints a summary of the mining results.
    """
    def __init__(self, file, minSup, sep):

        self.file = file
        self.minSup = minSup
        self.sep = sep
        self.transactions = []
        self.patterns = {}
        self.runtime = 0.0

    def read_file(self):
        # Read the CSV file
        lines = []
        with open(self.file, 'r') as f:
            lines = f.readlines()
            
        lines = [line.strip().split(self.sep) for line in lines]
        self.item_counts = Counter()
        for line in lines:  
            self.item_counts.update(line)
        for k,v in list(self.item_counts.items()):
            if v < self.minSup:
                del self.item_counts[k]
        self.item_counts = dict(sorted(self.item_counts.items(), key=lambda item: item[1], reverse=True))
        
        for line in lines:
            self.transactions.append([sorted([item for item in line if item in self.item_counts], key=lambda item: self.item_counts[item], reverse=True), 1])
            
            
    def _recursive_mine(self, root: 'Node', item_nodes: Dict[Any, Tuple[set, int]]) -> None:
        
        item_nodes = {k:v for k,v in sorted(item_nodes.items(), key=lambda item: item[1][1], reverse=True) if v[1] >= self.minSup}
        
        for item, (nodes, count) in item_nodes.items():
            newRoot = Node(root.item + [item], 0, None)
            
            self.patterns[tuple(newRoot.item)] = count
            newItemNodes = {}
            
            itemConuts = {}
            transactions = {}
            
            for node in nodes:
                transaction, count = node.traverse()
                if len(transaction) == 0:
                    continue
                transaction = tuple(transaction)
                if transaction not in transactions:
                    transactions[transaction] = 0
                transactions[transaction] += count
                
                for item in transaction:
                    if item not in itemConuts:
                        itemConuts[item] = 0
                    itemConuts[item] += count
                    
            itemConuts = {k:v for k,v in sorted(itemConuts.items(), key=lambda item: item[1], reverse=True) if v >= self.minSup}
            if len(itemConuts) == 0:
                continue
            
            for transaction, count in transactions.items():
                newTransaction = sorted([item for item in transaction if item in itemConuts], key=lambda item: itemConuts[item], reverse=True)
                curr = newRoot
                for item in newTransaction:
                    curr = curr.add_child(item, count)
                    if item not in newItemNodes:
                        newItemNodes[item] = [set(), 0]
                    newItemNodes[item][0].add(curr)
                    newItemNodes[item][1] += count
                    
            if len(newItemNodes) > 0:
                self._recursive_mine(newRoot, newItemNodes)
            
                    
            
    def mine(self) -> None:
        """Execute the FP-Growth algorithm to mine frequent patterns."""
        start_time = time.time()

        self.read_file()

        root = Node([], 0, None)
        item_nodes = {}
        for transaction, count in self.transactions:
            node = root
            for item in transaction:
                node = node.add_child(item, count)
                if item not in item_nodes:
                    item_nodes[item] = [set(), 0]
                item_nodes[item][0].add(node)
                item_nodes[item][1] += count
        
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
        
        
if __name__ == "__main__":
    file = "/home/tarun/cuPAMI/datasets/Transactional_pumsb.csv"
    sep = '\t'
    minSup = 38000
    outFile = "patterns.txt"
    
    # obj = FPGrowth(file, minSup, sep)
    # obj.mine()
    # obj.printResults()
    # obj.save_patterns(outFile, sep)
    
    
    # from PAMI.frequentPattern.basic.FPGrowth import FPGrowth as FPGrowth2
    
    # obj2 = FPGrowth2(file, minSup, sep)
    # obj2.mine()
    # obj2.printResults()
    # obj2.save(outFile)


class ParallelFPGrowth(FPGrowth):
    def _parallel_recursive_mine(self, args) -> Dict[Tuple[Any, ...], int]:
        """Helper function for parallel mining."""
        root, item_nodes, minSup = args
        local_patterns = {}
        item_nodes = {k: v for k, v in sorted(item_nodes.items(), key=lambda item: item[1][1], reverse=True) if v[1] >= minSup}
        
        for item, (nodes, count) in item_nodes.items():
            newRoot = Node(root.item + [item], 0, None)
            local_patterns[tuple(newRoot.item)] = count
            newItemNodes = {}
            itemCounts = {}
            transactions = {}

            for node in nodes:
                transaction, count = node.traverse()
                if len(transaction) == 0:
                    continue
                transaction = tuple(transaction)
                if transaction not in transactions:
                    transactions[transaction] = 0
                transactions[transaction] += count

                for item in transaction:
                    if item not in itemCounts:
                        itemCounts[item] = 0
                    itemCounts[item] += count

            itemCounts = {k: v for k, v in sorted(itemCounts.items(), key=lambda item: item[1], reverse=True) if v >= minSup}
            if len(itemCounts) == 0:
                continue

            for transaction, count in transactions.items():
                newTransaction = sorted([item for item in transaction if item in itemCounts], key=lambda item: itemCounts[item], reverse=True)
                curr = newRoot
                for item in newTransaction:
                    curr = curr.add_child(item, count)
                    if item not in newItemNodes:
                        newItemNodes[item] = [set(), 0]
                    newItemNodes[item][0].add(curr)
                    newItemNodes[item][1] += count

            if len(newItemNodes) > 0:
                sub_patterns = self._parallel_recursive_mine((newRoot, newItemNodes, minSup))
                local_patterns.update(sub_patterns)
        
        return local_patterns

    def _recursive_mine_parallel(self, root: 'Node', item_nodes: Dict[Any, Tuple[set, int]]) -> None:
        """Execute the recursive mining process in parallel."""
        processes = min(os.cpu_count(), len(item_nodes))
        with Pool(processes) as pool:
            chunk_size = len(item_nodes) // processes
            chunks = list(item_nodes.items())
            args = [
                (root, dict(chunks[i:i + chunk_size]), self.minSup)
                for i in range(0, len(chunks), chunk_size)
            ]
            results = pool.map(self._parallel_recursive_mine, args)

        for result in results:
            self.patterns.update(result)

    def mine(self) -> None:
        """Execute the FP-Growth algorithm to mine frequent patterns with parallelization."""
        start_time = time.time()

        self.read_file()

        root = Node([], 0, None)
        item_nodes = {}
        for transaction, count in self.transactions:
            node = root
            for item in transaction:
                node = node.add_child(item, count)
                if item not in item_nodes:
                    item_nodes[item] = [set(), 0]
                item_nodes[item][0].add(node)
                item_nodes[item][1] += count

        self._recursive_mine_parallel(root, item_nodes)
        self.runtime = time.time() - start_time


if __name__ == "__main__":
    file = "/home/tarun/cuPAMI/datasets/Transactional_pumsb.csv"
    sep = '\t'
    minSup = 38000
    outFile = "patterns.txt"
    
    # obj = ParallelFPGrowth(file, minSup, sep)
    # obj.mine()
    # obj.printResults()
    

    
    # file = "/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.csv"
    # sep = '\t'
    # minSup = 10
    
    
    obj = FPGrowth(file, minSup, sep)
    obj.mine()
    obj.printResults()
    obj.save_patterns(outFile, sep)
    
    
    # obj = ParallelFPGrowth(file, minSup, sep)
    # obj.mine()
    # obj.printResults()