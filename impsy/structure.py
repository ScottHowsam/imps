import numpy as np
from typing import List, Tuple

class Node:
    def __init__(self, output: np.ndarray, parent=None):
        self.output = output
        self.parent = parent
        self.children = []

class PredictionTree:
    def __init__(self, max_depth: int = 2, branching_factor: int = 3):
        self.root = None
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.best_branch = (None, float('-inf'))

    def build_tree(self, memory: List[np.ndarray], predict_function):
        # Reset best branch
        self.best_branch = (None, float('-inf'))
        # Create nodes for the entire memory
        nodes = [Node(output) for output in memory]
        for i in range(1, len(nodes)):
            nodes[i].parent = nodes[i-1]
            nodes[i-1].children.append(nodes[i])
        
        self.root = nodes[0]
        
        # Start building the tree from the last memory node
        self._recursive_build(nodes[-1], 0, memory, predict_function)

    def _recursive_build(self, node, depth, current_memory, predict_function):
        if depth >= self.max_depth:
            return
        
        for _ in range(self.branching_factor):
            last_memory = current_memory[-1]
            output = predict_function(last_memory)
            child = Node(output, parent=node)
            node.children.append(child)
            
            new_memory = current_memory[1:] + [output]
            self._recursive_build(child, depth + 1, new_memory, predict_function)

    def extract_branch(self, node: Node) -> np.ndarray:
        branch = []
        current = node
        while current:
            branch.append(current.output)
            current = current.parent
        return np.array(branch[::-1])

    def rank_branches(self, heuristic_function) -> List[Tuple[np.ndarray, float]]:
        branches = []
        
        def dfs(node):
            if not node.children:
                branch = self.extract_branch(node)
                heuristic_value = heuristic_function(branch)
                branches.append((branch, heuristic_value))
                if heuristic_value > self.best_branch[1]:
                    self.best_branch = (branch, heuristic_value)
                return
            
            for child in node.children:
                dfs(child)
        
        dfs(self.root)
        return branches

# Predict next from previous output
def predict_next(output: np.ndarray) -> np.ndarray:
    # This is a placeholder for your RNN prediction model
    # Replace this with your actual prediction function
    return np.random.randint(0, 128, size=4)  # Returning prediction as an array of 4 integers

# Initialize random memory
memory = [np.random.randint(0, 128, size=4) for _ in range(3)]

# Create and build the prediction tree
tree = PredictionTree(max_depth=2, branching_factor=2)
tree.build_tree(memory, predict_next)

# Sample branches
from impsy.heuristics import rhythmic_consistency
sampled_branches = tree.rank_branches(rhythmic_consistency)

# Print the first few sampled branches
for i, (branch, heuristic_value) in enumerate(sampled_branches):
    print(f"Branch {i + 1} Heuristic Value: {heuristic_value:.4f}):")
    print(branch)
    print(f"Branch length: {len(branch)}")
    print()