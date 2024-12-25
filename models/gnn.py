"""
    --- Graph Neural Network (GNN) Encoder ---
    Graph community encoder uses graph embeddings output from an graph encoder to estimate community memberships of nodes.
"""

from typing import Union, List, Optional

import gc
from functools import partial

import torch
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, Module, functional as F, ModuleList, Softmax, ModuleList
from torch_geometric.typing import Adj

# Base class for graph nn encoder
class GraphEncoder(Module):
    r""" A graph neural network encoder that encodes graph data into node embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, edge_feat_dim: int,
                 gnn_type: str, norm_type: str, act_type: str, enc_concat: bool = False,
                 dropout: float = 0.0):
        super(GraphEncoder, self).__init__()

        assert gnn_type in ['GCNConv', 'GATConv', 'SAGEConv', 'GraphConv', 'NNConv']
        assert norm_type in ['BatchNorm', 'LayerNorm', 'GraphNorm', None]
        assert act_type in ['selu', 'relu', 'leaky_relu', None]

        self.gnn = getattr(pyg_nn, gnn_type)
        self.norm = getattr(pyg_nn, norm_type) if norm_type != None else None
        self.act = partial(getattr(F, act_type), inplace=True) if act_type != None else lambda x: x

        self.norm_transfer_args = "x -> x" if norm_type == 'BatchNorm' else "x, batch -> x"

        assert num_layers > 0
        assert input_dim > 0
        assert output_dim > 0
        assert hidden_dim > 0

        self.edge_feat_dim = edge_feat_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Concatenate embeddings from all layers
        self.enc_concat = enc_concat

        # Build graph convolutional blocks
        self.conv_blocks = self.build_blocks()

    def build_seq_conv_layers(self):
        # Build a sequence of graph convolutional layers
        # Each layer is followed by an activation function and a normalization layer
        conv_layers = ModuleList()
        for i in range(self.num_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            output_dim = self.output_dim if i == self.num_layers - 1 else self.hidden_dim
            conv_layers.append((self.gnn(input_dim, output_dim, droput=self.dropout),
                                "x, edge_index, edge_attr -> x"))
            if i < self.num_layers - 1 and self.norm != None:
                conv_layers.append((self.norm(output_dim), self.norm_transfer_args))
            conv_layers.append(self.act)
        return conv_layers

    def build_seq_nnconv_layers(self):
        # Build a sequence of neural network convolutional layers
        # Each layer is followed by an activation function and a normalization layer
        conv_layers = ModuleList()
        for i in range(self.num_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            output_dim = self.output_dim if i == self.num_layers - 1 else self.hidden_dim
            nn = Sequential(Linear(self.edge_feat_dim, input_dim * output_dim), ReLU(inplace=True))
            conv_layers.append((self.gnn(input_dim, output_dim, nn=nn, aggr='mean'),
                                "x, edge_index, edge_attr -> x"))
            if i < self.num_layers - 1 and self.norm != None:
                conv_layers.append((self.norm(output_dim), self.norm_transfer_args))
            conv_layers.append(self.act)
        return conv_layers

    def build_blocks(self):
        # Build a sequence of graph convolutional layers
        # Each layer is followed by an activation function and a normalization layer
        blocks = ModuleList()
        for i in range(self.num_layers):
            layers = []
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            output_dim = self.output_dim if i == self.num_layers - 1 else self.hidden_dim
            if self.gnn.__name__ == 'NNConv':
                nn = Sequential(Linear(self.edge_feat_dim, input_dim * output_dim), ReLU(inplace=True))
                layers.append((self.gnn(input_dim, output_dim, nn=nn, aggr='mean'),
                                "x, edge_index, edge_attr -> x"))
            else:
                layers.append((self.gnn(input_dim, output_dim, droput=self.dropout),
                               "x, edge_index, edge_attr -> x"))
            if i < self.num_layers - 1 and self.norm != None:
                layers.append((self.norm(output_dim), self.norm_transfer_args))
            layers.append(self.act)
            blocks.append(pyg_nn.Sequential("x, edge_index, edge_attr, batch", layers))
        return blocks

    # TODO: Implement block-wise graph encoder like in Dual-HINet that concatenates embeddings
    def build_conv_blocks(self):
        r""""""
        return NotImplementedError

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, batch: Tensor) -> Tensor:
        r"""
        Forward pass of the graph encoder.
        :param x: Node features
                Shape (b*n,f)
        :param edge_index: Edge index tensor
                Shape (2, e)
        :param edge_attr: Edge attribute tensor
                Shape (e, f)
        :param batch: Batch tensor
                Shape (n,)
        :return: Encoded node embeddings
                Shape (b*n, d) or (b*n, d*l) or (b*n, d)
        """
        z_list = []
        for layer in self.conv_blocks:
            x = layer(x, edge_index, edge_attr, batch)
            z_list.append(x)
        if self.enc_concat:
            z = torch.cat(z_list, dim=-1)
        else:
            z = z_list[-1]
        return z  # [B * N, D * L] or [B * N, D]

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
