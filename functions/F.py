"""
    This module contains the functions for tensor operations and other utilities.

"""
import torch
from torch import Tensor
from torch.nn.functional import softmax

from typing import Union

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
        adj = adj.unsqueeze(-1)  # [B, C, C, 1]
    adj = adj.permute(-4, -1, -3, -2)  # [B, L, C, C]
    # TODO: handle dimensions for matmul
    # Cluster node features (S^T * X)
    x_clustered = torch.matmul(s.transpose(1, 2), x)
    # Cluster adjacency matrix (S^T * A * S)
    s = s.unsqueeze(-3)  # [B, 1, N, C]
    adj_clustered = torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...ik->...jk', s, adj), s)
    adj_clustered = adj_clustered.permute(-4, -2, -1, -3)  # [B, C, C, L]
    if adj_clustered.size(-1) == 1:
        adj_clustered = adj_clustered.squeeze(-1)
    return x_clustered, adj_clustered

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


if __name__ == "__main__":
    # test cluster_graph
    # trial 1
    x = torch.rand(16, 45, 8)
    adj = torch.rand(16, 45, 45)
    cluster_assignments = torch.rand(16, 45, 3)
    x_clustered1, adj_clustered1 = cluster_graph(x, adj, cluster_assignments)
    print(x_clustered1.size(), adj_clustered1.size())
