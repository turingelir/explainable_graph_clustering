"""
    --- Graph Neural Network (GNN) Encoder ---
    Graph community encoder uses graph embeddings output from an graph encoder to estimate community memberships of nodes.
"""

from typing import Union, List, Optional

import gc
from functools import partial

import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, Module, functional as F, ModuleList, Softmax, ModuleList

# Graph Convolutional Network (GCN) encoder
class GCN(Module):
    r"""
        Graph Convolutional Network (GCN) layer.
        
        The GCN layer is defined as:
            Z = act(D^(-1/2) * A * D^(-1/2) * X * W)
        Args:
            :param in_features: (int) The number of input features.
            :param out_features: (int) The number of output features.
            :param bias: (bool) If set to False, the layer will not learn an additive bias.
            :param activation: (Module) The activation function to use.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: Optional[Module] = None):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        r"""
            Forward pass of the GCN layer.
            Z = act(D^(-1/2) * A * D^(-1/2) * X * W)

            Args:
                :param x: (Tensor) The input tensor of shape [B, N, F] where B is the batch size, N is the number of nodes, 
                        and F is the number of input features.
                :param adj: (Tensor) The adjacency matrix of the graph of shape [B, N, N] where B is the batch size and 
                        N is the number of nodes.
            Returns:
                :return: (Tensor) The output tensor of shape [B, N, out_features] where out_features is the number of output features.
        """
        # Compute the degree matrix
        degree = torch.sum(adj, dim=-1)
        # Compute the degree matrix inverse square root
        degree_inv = torch.pow(degree + 1e-6, -0.5)
        # Create a diagonal matrix from the degree matrix
        d = torch.diag_embed(degree_inv)

        # Compute the normalized adjacency matrix
        adj_norm = torch.matmul(torch.matmul(d, adj), d)

        # Message passing
        z = torch.matmul(adj_norm, x)

        # Linear transformation
        z = torch.matmul(z, self.weight)

        # Add bias 
        if self.bias is not None:
            z = z + self.bias

        # Activation layer
        if self.act is not None:
            z = self.act(z)

        return z

class GNN(Module):
    r"""
        Graph Neural Network (GNN) block.
        For given graph convolutional layers, the GNN block applies the GCN layers in sequence.
        Batch normalization may be applied before each graph convolution layer.
        Residual skip connections may be applied to the output of each graph convolution layer.
        Or skip concatenation connections may be applied to the output of each graph convolution layer.

        Args:
            :param in_features: (int) The number of input features.
            :param out_features: (int) The number of output features.
            :param num_layers: (int) The number of graph convolutional layers.
            :param bias: (bool) If set to False, the layer will not learn an additive bias.
            :param activation: (Module) The activation function to use.
            :param batch_norm: (bool) If set to True, batch normalization will be applied before each graph convolution layer.
            :param residual: (bool) If set to True, residual skip connections will be applied to the output of each graph convolution layer.
            :param skip_concat: (bool) If set to True, skip concatenation connections will be applied to the output of each graph convolution layer.
    """
    def __init__(self, in_features: int, out_features: int, num_layers: int, bias: bool = True, activation: Optional[Module] = None,
                 GCN: Module = GCN, batch_norm: bool = False, residual: bool = False, skip_concat: bool = False):
        super(GNN, self).__init__()
        # Graph convolutional layer parameters
        self.in_features = in_features
        self.out_features = out_features
        # Number of graph convolutional layers in the block
        self.num_layers = num_layers
        # Bias
        self.bias = bias
        # Activation function
        self.activation = activation
        # Batch normalization
        self.batch_norm = batch_norm
        # NOTE: Residual and skip concatenation connections are mutually 
        assert not (residual and skip_concat), "Residual and skip concatenation connections are mutually exclusive."
        # Residual skip connections
        self.residual = residual
        # Skip concatenation connections
        self.skip_concat = skip_concat
        

        # Initialize the graph convolutional layers
        self.layers = ModuleList()
        for _ in range(num_layers-1):
            # Batch normalization
            if batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(in_features))
            # Graph convolutional layer
            if self.skip_concat:
                in_features = in_features + out_features
            self.layers.append(GCN(in_features, out_features, bias, activation))

        # Last graph convolutional layer
        if batch_norm:
            self.layers.append(torch.nn.BatchNorm1d(in_features))
        self.layers.append(GCN(in_features, out_features, bias))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        r"""
            Forward pass of the GNN block.
            Args:
                :param x: (Tensor) The input tensor of shape [B, N, F] where B is the batch size, N is the number of nodes, 
                        and F is the number of input features.
                :param adj: (Tensor) The adjacency matrix of the graph of shape [B, N, N] where B is the batch size and 
                        N is the number of nodes.
            Returns:
                :return: (Tensor) The output tensor of shape [B, N, out_features] where out_features is the number of output features.
        """
        # Apply the graph convolutional layers
        for layer in self.layers:
            if layer.__class__.__name__ == 'BatchNorm1d':
                # Apply batch normalization
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1) # Permute to [B, F, N] for batch normalization
            else:
                z = layer(x, adj)
                if self.residual:
                    z = z + x
                elif self.skip_concat:
                    z = torch.cat((z, x), dim=-1)
                x = z
        return x


