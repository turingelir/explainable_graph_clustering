"""
    TODO:
        - Args and return types for all functions
        ~ Base class for graph community based models
        - Class for community membership oracle (or not)

"""
from typing import Union, List, Optional

from torch import Tensor
from torch.nn import Module
import torch_geometric.nn as pyg_nn

"""
    Graph community encoder uses graph embeddings output from an graph encoder to estimate community memberships of nodes.
"""


# Loss function's for
class GraphCommunityPooling(Module):
    r""" A graph pooling layer that estimates community memberships of nodes using graph embeddings.
    """

    def __init__(self, channels: Union[int, List[int]], num_communities: int, community_mapping_type: str = 'soft',
                 dropout: float = 0.0, return_comm_memberships: bool = False):
        super(GraphCommunityPooling, self).__init__()

        # Channels are used for the MLP layer
        channels = channels if isinstance(channels, list) else [channels]

        self.mlp = pyg_nn.MLP(channels + [num_communities], batch_norm=False)
        self.dropout = dropout

        self.community_mapping_type = community_mapping_type if community_mapping_type in ['soft', 'hard'] else 'soft'
        self.return_comm_memberships = return_comm_memberships

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor,
                mask: Optional[Tensor] = None):
        pass
