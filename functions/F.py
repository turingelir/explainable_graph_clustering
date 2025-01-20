"""
    This module contains the functions for tensor operations and other utilities.

"""
import torch
from torch import Tensor
from torch.nn.functional import softmax
from graspologic.simulations import sbm

from typing import Union, Tuple, List

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def to_hard_assignment(p: Tensor) -> Tensor:
    r""" Hard assignment function for Softmax soft probability input tensor.
        This way a differentiable assignment can be converted to a hard assignment.
        Args:
            :arg p: Tensor of probabilities of shape [B, N, C]. (Soft assignment from Softmax)
        Returns:
            :return h: Hard assignment tensor of shape [B, N, C]. (1 for max, 0 for others)
    """
    m = p.max(dim=-1, keepdim=True)[0] - 1e-6
    h = (p > m).float()
    return (h - p).detach() + p


def cluster_graph(x: Tensor, adj: Tensor, s: Tensor) -> Union[Tensor, Tensor]:
    r""" Cluster graph function for graph data.
    Args:
        :arg x: Node feature tensor of shape [B, N, F].
        :arg adj: Adjacency matrix tensor of shape [B, N, N] or [B, N, N, L].
        :arg s: Cluster assignment tensor of shape [B, N, C].
    Returns:
        :return x_clustered: Clustered node feature tensor of shape [B, N, C].
        :return adj_clustered: Clustered adjacency matrix tensor of shape [B, C, C] or [B, C, C, L].
    """
    assert x.dim() == 3, "Node feature tensor must be 3-dimensional."
    assert adj.dim() == 3 or adj.dim() == 4, "Adjacency matrix tensor must be 3 or 4-dimensional."
    assert s.dim() == 3, "Cluster assignment tensor must be 3-dimensional."
    assert x.size(0) == adj.size(0) == s.size(0), "Batch size mismatch."
    assert x.size(1) == adj.size(1) == s.size(1), "Node size mismatch."
    if adj.dim() == 3:
        adj = adj.unsqueeze(-1)  # [B, N, N, 1]
    adj = adj.permute(-4, -1, -3, -2)  # [B, L, N, N]
    # TODO: handle dimensions for matmul
    # Cluster node features (S^T * X)
    x_clustered = torch.matmul(s.transpose(1, 2), x)
    # Cluster adjacency matrix (S^T * A * S)
    s = s.unsqueeze(-3)  # [B, 1, N, C]
    adj_clustered = torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...ik->...jk', s, adj), s) # [B, L, C, C]
    adj_clustered = adj_clustered.permute(-4, -2, -1, -3)  # [B, C, C, L]
    if adj_clustered.size(-1) == 1:
        adj_clustered = adj_clustered.squeeze(-1)
    return x_clustered, adj_clustered

def cluster_edges(adj: Tensor, s: Tensor) -> Tensor:
    r""" Cluster edges function for graph data.
    Args:
        :arg adj: Adjacency matrix tensor of shape [B, N, N] or [B, N, N, L].
        :arg s: Cluster assignment tensor of shape [B, N, C].
    Returns:
        :return adj_clustered: Clustered adjacency matrix tensor of shape [B, C, C, L].
    """
    assert adj.dim() == 3 or adj.dim() == 4, "Adjacency matrix tensor must be 3 or 4-dimensional."
    assert s.dim() == 3, "Cluster assignment tensor must be 3-dimensional."
    assert adj.size(0) == s.size(0), "Batch size mismatch."
    assert adj.size(1) == s.size(1), "Node size mismatch."
    if adj.dim() == 3:
        adj = adj.unsqueeze(-1)  # [B, N, N, 1]
    adj = adj.permute(-4, -1, -3, -2)  # [B, L, N, N]
    s = s.unsqueeze(-3)  # [B, 1, N, C]
    adj_clustered = torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...ik->...jk', s, adj), s) # [B, L, C, C]
    adj_clustered = adj_clustered.permute(-4, -2, -1, -3)  # [B, C, C, L]
    if adj.dim() == 3:
        adj_clustered = adj_clustered.squeeze(-1)
    return adj_clustered

def softmax_w_temperature(x: Tensor, dim: int = -1, temperature: float = 1.0) -> Tensor:
    r""" Softmax function with temperature scaling.
    Args:
        :arg x: Input tensor of shape [*, D].
        :arg dim: Dimension to apply softmax.
        :arg temperature: Temperature scaling factor.
    Returns:
        :return p: Softmax probability tensor of shape [*, D].
    """
    p = softmax(x / temperature, dim=dim)
    return p

def rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)

def rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))
    return out

def generate_graph(n_samples: int=1, n_nodes: int=300, n_clusters: int=3, 
                   cluster_counts: List[int]=None, conn_prob: List[List[float]]=None,
                   node_features: int=2, cluster_means: List[List[float]]=None, 
                   cluster_covs: List[List[List[float]]]=None,
                   random_order: bool=True
                   ) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
        Generate graph data randomly.
        Args:
            :arg n_samples: Number of samples to generate.
            :arg n_nodes: Number of nodes in the graph.
            :arg n_clusters: Number of clusters in the graph.
            :arg cluster_counts: Number of nodes in each cluster.
            :arg conn_prob: Connectivity probability between clusters.
            :arg node_features: Number of features for each node.
            :arg cluster_means: Mean values for each cluster.
            :arg cluster_covs: Covariance matrices for each cluster.
            :arg random_order: Randomize cluster assignments.
        Returns:
            :return nodes: Node feature tensor of shape [B, N, F].
            :return graph: Adjacency matrix tensor of shape [B, N, N].
            :return partition: Cluster assignment tensor of shape [B, N, C].
    """
    ## Create example graph and partition
    # batch, number_of_nodes, number_of_clusters
    b, n_nodes, n_clusters = n_samples, n_nodes, n_clusters
    if b != 1:
        raise NotImplementedError("Batch size must be 1.")
    
    ## Edges: A
    # Create graphs from a stochastic block model (SBM)
    n = [n_nodes // n_clusters] * n_clusters if cluster_counts is None else cluster_counts
    p = [[0.6, 0.2, 0.3], [0.2, 0.65, 0.2], [0.3, 0.2, 0.7]] if conn_prob is None else conn_prob

    # Print out the graph and community memberships
    print('Graph # of nodes and p values per community:')
    print('n:', n)
    print('p:', p)

    graph, community_memberships = sbm(n=n, p=p, loops=False, return_labels=True)

    ## Cluster assignments: S
    # Randomize the memberships order
    community_memberships = torch.tensor(community_memberships).unsqueeze(0)
    if random_order:
        community_memberships = community_memberships[:,torch.randperm(sum(n))]
    # Convert to one-hot encoding
    partition = torch.functional.F.one_hot(community_memberships, num_classes=n_clusters).float()

    ## Nodes: X
    cluster_means = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]] if cluster_means is None else cluster_means
    if cluster_covs is None:
        cluster_covs = [
            [[0.2, 0.1], [0.1, 0.2]],  # Covariance matrix for cluster 1
            [[0.3, 0.2], [0.2, 0.3]],  # Covariance matrix for cluster 2
            [[0.4, 0.3], [0.3, 0.4]]   # Covariance matrix for cluster 3
        ]
    # Generate node features
    nodes = torch.zeros(sum(n), node_features)
    cluster_distr = torch.distributions.MultivariateNormal
    for i in range(n_clusters):
        cluster_mean = torch.tensor(cluster_means[i])
        cluster_cov = torch.tensor(cluster_covs[i])
        mask = community_memberships[0] == i
        nodes[mask] = cluster_distr(cluster_mean, cluster_cov).sample([mask.sum()])
    ### Return tensors (A, X, S)
    # Graph and node tensors (single graph sample)
    graph = torch.tensor(graph, dtype=torch.float32).unsqueeze(0)
    # nodes = torch.eye(sum(n)).unsqueeze(0)
    nodes = nodes.unsqueeze(0)
    partition = partition.unsqueeze(0)

    return nodes, graph, partition
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate graph data
    graph, nodes, partition = generate_graph()

    # Display shapes
    print("Graph shape:", graph.shape)
    print("Nodes shape:", nodes.shape)
    print("Partition shape:", partition.shape)

    # Display using show_mat function for graph

    from utils import show_mat
    show_mat(graph.squeeze(), "Graph", show=True, cmap='binary')

    # Scatter plot for node features with partition labels
    # Node-feature vectors are already 2 dimensional
    
    plt.scatter(nodes.squeeze()[:, 0], nodes.squeeze()[:, 1], c=partition.squeeze().argmax(dim=-1))
    plt.title("Node Features")
    plt.show()
