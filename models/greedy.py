"""
    --- Greedy optimization algorithm ---
    This file contains the general greedy optimization algorithm framework for graph clustering and community detection tasks.
    According to loss/likelihood functions, iterative algorithms are used to optimize the objective function.
    For example: Louvain algorithm is a popular greedy algorithm for community detection.
"""
import random
import torch

# Optimizer class
# Objective function will be taken as input
# It is a function that takes a graph and a partition and returns a scalar value that represents the quality of the partition
# Iterative method where nodes are moved between clusters will be used
# A gain calculation method will be used to determine the best move
# The algorithm will be run until no further improvement can be made 
# Best partition will be returned
# Graphs are represented as adjacency matrices [n x n] where n is the number of nodes
# Clusters are represented as one-hot encoded vectors [n x c] where c is the number of clusters
class GreedyOptimizer:
    def __init__(self, objective_function, random_order=False, seed=None, obj='max'):
        self.objective_function = objective_function

        self.best_partition = None
        self.best_score = None

        self.random_order = random_order
        self.seed = seed

        if obj == 'max':
            self.gain = lambda a, b: a < b
        elif obj == 'min':
            self.gain = lambda a, b: a > b
        else:
            raise ValueError(f"Invalid objective function: {obj}. Expected 'max' or 'min'.")

    def initialize(self, graph, partition):
        self.best_partition = partition
        self.best_score = self.objective_function(graph, partition)

    def clear(self):
        self.best_partition = None
        self.best_score = None

    def optimize(self, graph, partition):
        self.initialize(graph, partition)

        # Iterative optimization
        # Move nodes between clusters until no further improvement can be made
        # FIXME: FOR DEBUGGING PURPOSES
        j = 0
        while True:
            # Order of nodes to move between clusters
            order = list(range(graph.size(1))) # [0, 1, 2, ..., n]
            if self.random_order:
                random.seed(self.seed)
                random.shuffle(order)
            # Iterate over the nodes
            for i in order:
                # FIXME: FOR DEBUGGING PURPOSES
                print(f"Node step: {i+j*graph.size(1)}")
                # Get the best move for a node to switch clusters
                move = self.get_best_move(graph, partition, i)
                # If a move is possible
                if move is not None:
                    # Move the node by updating the partition
                    partition = self.move_node(partition, move)
                    # Evaluate the new partition
                    score = self.objective_function(graph, partition)
                    # If the new partition is better than the current best partition
                    if self.gain(score, self.best_score):
                        # Update the best partition and score
                        self.best_partition = partition
                        self.best_score = score
                        break
                    else:
                        # If the new partition is not better, revert the move
                        partition = self.best_partition
                        # Failure of best move method is not expected
                        raise Exception("I WAS WRONG. Best move method did NOT return a better move.")
                else:
                    # If no move is possible, break the loop
                    break
            # If no move is possible, break the loop
            if move is None:
                break
        return self.best_partition
    
    def get_best_move(self, graph:torch.Tensor, partition:torch.Tensor, node:int)->torch.Tensor:
        r"""
            Get the best move for a node in the graph.
            Args:
                :param graph: (Tensor) The graph adjacency matrix.
                        Shape is [B, N, N] where N is the number of nodes, and B is the batch size.
                :param partition: (Tensor) The current partition of the nodes.
                        Shape is [B, N, C] where C is the number of clusters, and B is the batch size.
                :param node: (int) The node index to move.
            Returns:
                :return move: (Tensor) The best move for the node. 
                        Shape is [B, C] where C is the number of clusters, and B is the batch size.
                        Move is represented as a one-hot encoded vector of the cluster to move to.
        """
        # NOTE: While batch dimension is supported for operations, B is expected to be 1
        assert graph.size(0) == 1, f"Expected batch size of 1, got {graph.size(0)}. Batched graphs are yet to be supported."

        # Get the current cluster of the node
        current_cluster = partition[:, node] # [B, C]
        # Get the current score of the partition
        current_score = self.objective_function(graph, partition) # [B]
        # Store initial cluster assignment
        init_cluster = current_cluster.clone()
        # Initialize the best move and the best score per batch
        best_move = current_cluster # Best move is to stay in the same cluster, initially
        best_score = current_score

        order = list(range(partition.size(-1)))
        # Randomize the order of the clusters
        if self.random_order:
            random.seed(self.seed)
            random.shuffle(order)
        # Iterate over the clusters
        for i in order:
            # FIXME: FOR DEBUGGING PURPOSES
            print(f"Cluster step: {i}")
            # If the node is already in the cluster, skip
            if torch.all(current_cluster[:, i] == 1):
                continue
            # Create a move to the cluster
            move = torch.zeros_like(current_cluster)
            move[:, i] = 1
            # Create a new partition by moving the node to the cluster
            new_partition = self.move_node(partition, move, node)

            # Evaluate the new partition
            new_score = self.objective_function(graph, new_partition)
            # For batches where the new score is better
            mask = self.gain(new_score, best_score)

            # Update the best move and the best score
            best_move = torch.where(mask.unsqueeze(-1), new_partition[:, node], best_move)
            best_score = torch.where(mask, new_score, best_score)

        # Check if the best move is different from the initial cluster assignment
        if torch.all(best_move == init_cluster):
            return None
        
        return best_move
    
    def move_node(self, partition, move, node):
        r"""
            Move a node to a new cluster in the partition.
            Args:
                :param partition: (Tensor) The current partition of the nodes.
                        Shape is [B, N, C] where C is the number of clusters, and B is the batch size.
                :param move: (Tensor) The move to make.
                        Shape is [B, C] where C is the number of clusters, and B is the batch size.
                        Move is represented as a one-hot encoded vector of the cluster to move to.
                :param node: (int) The node index to move.
            Returns:
                :return new_partition: (Tensor) The new partition of the nodes.
                        Shape is [B, N, C] where C is the number of clusters, and B is the batch size.
        """
        new_partition = partition.clone()
        new_partition[:, node] = move
        return new_partition

    
    def get_best_partition(self):
        return self.best_partition
    
    def get_best_score(self):
        return self.best_score


    
if __name__ == '__main__':
    # Test the greedy optimizer
    # Define a simple objective function
    def simple_objective(graph, partition):
        return torch.sum(graph @ partition, dim=(1, 2))
    
    # Define a simple graph and partition
    b, n, c = 1, 100, 3
    graph = torch.bernoulli(torch.zeros((b, n, n)).uniform_(0, 1))
    graph[:, torch.eye(n).bool()] = 0
    partition = torch.randint(0, c, (b, n))
    partition = torch.nn.functional.one_hot(partition, num_classes=c).float()
    
    # Initialize the greedy optimizer
    optimizer = GreedyOptimizer(simple_objective)
    
    # Optimize the partition
    best_partition = optimizer.optimize(graph, partition)
    
    # Print the best partition
    # print(best_partition)
    
    # Print the best score
    print(optimizer.get_best_score())
    