"""
    This module contains the objective loss and regularization functions used for graph clustering and community detection tasks.
    Main objective functions to be compared upon are:
        - Modularity loss
        - Min-cut loss
        - SBM likelihood estimation
        - DC-SBM likelihood estimation
        ...
        - Entropy (?)
        - Energy based approaches (?)

"""
import torch
from torch import Tensor
from typing import Tuple


def joint_loss(losses: dict):
    r"""Compute joint loss by summing dict of losses while applying weights."""
    return sum([loss['weight'] * loss['result'] for loss in losses.values()])

def modularity_loss(init_adj: Tensor, new_adj: Tensor, assign_mat: Tensor, reduction: str = 'mean') -> Tensor:
    r"""
        Calculate the spectral loss of community memberships for the multi-graph, according to the modularity metric.
        Args:
            :param init_adj: (Tensor) The initial adjacency matrix of the graph.
                    Shape is [B, N, N, L] or [B, N, N] where B is the batch size, N is the number of nodes, 
                    and L is the number of layers of a multi-graph.
            :param new_adj: (Tensor) The new adjacency matrix of the graph after the community assignment of edges.
                    Shape is [B, C, C, L] or [B, C, C] where B is the batch size, C is the number of communities,
                    and L is the number of layers of a multi-graph.
            :param assign_mat: (Tensor) The community assignment of nodes.
                    Shape is [B, N, C] where B is the batch size, N is the number of nodes, and C is the number of communities.
            :param reduction: (str) The reduction method to use for the modularity loss.
        Returns:
            :return loss: (Tensor) The mean modularity loss across the batch.
    """
    assert init_adj.dim() in [3, 4], f"Expected input to have 3 or 4 dimensions, got {init_adj.dim()}"
    assert init_adj.dim() != 4 or init_adj.shape[-1] == new_adj.shape[-1], (f"Expected input and target to have last dimension "
                                                     f"equal, got {init_adj.shape[-1]} and {new_adj.shape[-1]}")
    assert assign_mat.dim() == 3, f"Expected community assignment to have 3 dimensions, got {assign_mat.dim()}"
    assert init_adj.shape[-2] == assign_mat.shape[-2], (
        f"Expected input and community assignment to have second-to-last dimension "
        f"equal, got {init_adj.shape[-2]} and {assign_mat.shape[-2]}")
    assert new_adj.shape[-2] == assign_mat.shape[-1], (
        f"Expected input and community assignment to have second-to-last dimension "
        f"equal, got {init_adj.shape[-2]} and {assign_mat.shape[-2]}")
    if init_adj.dim() == 3:
        init_adj = init_adj.unsqueeze(-1)
    if new_adj.dim() == 3:
        new_adj = new_adj.unsqueeze(-1)  # [B, C, C, 1]
    new_adj = new_adj.permute(-4, -1, -3, -2)  # [B, L, C, C]

    # Compute the node_strength and layer-wise sum of edges of the initial adjacency matrix
    node_strength = torch.sum(init_adj, dim=-2).transpose(-2, -1)  # [B, L, N]
    m = torch.sum(node_strength, dim=-1)  # sum of all edges for each graph layer  -> [B, L]
    # 
    ca = torch.einsum('bij,bki->bkj', assign_mat, node_strength).unsqueeze(-1)  # [B, L, C, 1]
    cb = torch.einsum('bij,bjk->bik', node_strength, assign_mat).unsqueeze(-2)  # [B, L, 1, C]

    normalizer = torch.einsum('...ij,...jk->...ik', ca, cb) / 2 / m.view(m.shape + (1, 1))  # [B, L, C, C]
    decompose = new_adj - normalizer
    spectral_loss = -torch.einsum('...ii', decompose) / 2 / m  # [B, L]

    spectral_loss = torch.mean(spectral_loss, dim=-1)  # Mean loss across the graph-layers

    if reduction == 'mean':    # Mean loss across the batch
        return torch.mean(spectral_loss)
    elif reduction == 'sum':
        return torch.sum(spectral_loss)
    elif reduction == 'none':
        return spectral_loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}. Expected 'mean', 'sum', or 'none'.")

def min_cut_loss(init_adj: Tensor, new_adj: Tensor, assign_mat: Tensor, reduction: str = 'mean') -> Tensor:
    return NotImplementedError

def clustering_regularization(assign_mat: Tensor, reduction: str = 'mean') -> Tuple[Tensor, Tensor]:
    r"""
        Compute clustering regularization losses called in one function for the community assignment matrix.
        Args:
            :param assign_mat: (Tensor) The community assignment of nodes.
                    Shape is [B, N, C] where B is the batch size, N is the number of nodes, and C is the number of communities.
            :param reduction: (str) The reduction method to use for the clustering regularization loss.
            Returns:
            :return o_loss: (Tensor) The mean orthogonality regularization loss across the batch.
            :return c_loss: (Tensor) The mean collapse regularization loss across the batch."""
    assert assign_mat.dim() == 3, f"Expected community assignment to have 3 dimensions, got {assign_mat.dim()}"
    ss = torch.einsum('bij,bik->bjk', assign_mat, assign_mat)  # S.T x S -> shape [B, C, C]
    o_loss = orthogonality_loss(assign_mat, ss=ss) # Orthogonality loss
    c_loss = collapse_loss(assign_mat, ss=ss, ) # Collapse 
    return o_loss, c_loss

def orthogonality_loss(assign_mat: Tensor, ss: Tensor = None, reduction: str = 'mean') -> Tensor:
    r"""
         Calculate the orthogonality regularization loss of the community assignment matrix.
        For avoiding degenerate local minima cases: assigning all nodes to the same cluster,
        and assigning all nodes with equal probability to all clusters.
        Args:
            :param assign_mat: (Tensor) The community assignment of nodes.
                    Shape is [B, N, C] where B is the batch size, N is the number of nodes, and C is the number of communities.
            :param ss: (Tensor) The dot product of the community assignment matrix.
                    Taken as parameter to avoid recomputation.
                    Shape is [B, C, C] where B is the batch size, and C is the number of communities.
            :param reduction: (str) The reduction method to use for the orthogonality loss.
        Returns:
            :return loss: (Tensor) The mean orthogonality regularization loss across the batch.
    """
    assert assign_mat.dim() == 3, f"Expected community assignment to have 3 dimensions, got {assign_mat.dim()}"
    if ss is None:
        ss = torch.einsum('bij,bik->bjk', assign_mat, assign_mat)  # S.T x S -> shape [B, C, C]
    i_s = torch.eye(ss.size(-1)).type_as(ss).unsqueeze(0)  # Identity matrix -> shape [1, C, C]
    ortho_loss = torch.linalg.norm(
        (ss / torch.linalg.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.linalg.norm(i_s)), dim=(-1, -2))  # [B]
    if reduction == 'mean':  # Mean loss across the batch
        return torch.mean(ortho_loss)
    elif reduction == 'sum':
        return torch.sum(ortho_loss)
    elif reduction == 'none':
        return ortho_loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}. Expected 'mean', 'sum', or 'none'.")

def collapse_loss(assign_mat: Tensor, ss: Tensor = None, reduction: str = 'mean') -> Tensor:
    r"""
         Calculate the collapse regularization loss of the community assignment matrix.
        It is used to avoid degenerate local minima: assigning all nodes to the same cluster.
        Args:
            :param assign_mat: (Tensor) The community assignment of nodes.
                    Shape is [B, N, C] where B is the batch size, N is the number of nodes, and C is the number of communities.
            :param ss: (Tensor) The dot product of the community assignment matrix.
                    Taken as parameter to avoid re-computation.
                    Shape is [B, C, C] where B is the batch size, and C is the number of communities.
            :param reduction: (str) The reduction method to use for the collapse loss.
        Returns:
            :return loss: (Tensor) The mean collapse regularization loss across the batch.
                        Normalized to range [0, √C].  
    """
    assert assign_mat.dim() == 3, f"Expected community assignment to have 3 dimensions, got {assign_mat.dim()}"
    if ss is None:
        ss = torch.einsum('bij,bik->bjk', assign_mat, assign_mat)  # S.T x S -> shape [B, C, C]
    num_nodes = assign_mat.size(-2)
    i_s = torch.eye(ss.size(-1)).type_as(ss).unsqueeze(0)  # Identity matrix -> shape [1, C, C]
    collapse_loss = torch.linalg.norm(torch.sum(ss, dim=-1, keepdim=True), dim=(-1, -2)) / num_nodes * \
                    torch.linalg.norm(i_s) - 1  # [B]
    if reduction == 'mean':  # Mean loss across the batch
        return torch.mean(collapse_loss)
    elif reduction == 'sum':
        return torch.sum(collapse_loss)
    elif reduction == 'none':
        return collapse_loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}. Expected 'mean', 'sum', or 'none'.")


if __name__ == "__main__":
    # Run some tests

    # Sample some tensors
    B, N, C = 16, 100, 10
    init_adj = torch.rand(B, N, N)
    new_adj = torch.rand(B, C, C)
    assign_mat = torch.rand(B, N, C)

    # Test modularity loss
    loss = modularity_loss(init_adj, new_adj, assign_mat, reduction='mean')
    print(f"Modularity loss: {loss}")

    loss_not_reduced = modularity_loss(init_adj, new_adj, assign_mat, reduction='none')
    print(f"Modularity loss (not reduced): {loss_not_reduced}")
