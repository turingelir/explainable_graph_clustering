"""
    --- Graph Neural Network (GNN) Encoder ---
    Graph community encoder uses graph embeddings output from an graph encoder to estimate community memberships of nodes.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Union, List, Optional

import gc
from functools import partial

import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, SELU, Module, functional as F, ModuleList, Softmax, ModuleList

from functions import to_hard_assignment

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
        Graph Neural Network (GNN) block that encodes graph data into node embeddings.
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
                # Update the input features for the next layer
                in_features = in_features + out_features
            # Add the graph convolutional layer
            self.layers.append(GCN(in_features, out_features, bias, activation))

            # Update the input features for the next layer
            in_features = out_features

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
                :return: (Tensor) The output tensor of shape [B, N, out_features] which are node embeddings for the graph.
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

class CommDetGNN(Module):
    r"""
        Graph Neural Network (GNN) community detection module.
        The Community Detection GNN uses graph embeddings output from an graph encoder to estimate community memberships of nodes.
    """ 
    def __init__(self, num_clusters: int, gnn: Module, mlp: Module,
                 hard_assignment: bool = False):
        super(CommDetGNN, self).__init__()
        # Number of target clusters
        self.num_clusters = num_clusters
        # Graph neural network encoder parameters
        self.gnn = gnn # GNN encoder
        # Community assignment MLP
        self.mlp = mlp # no activation function is applied
        assert mlp[0].in_features == gnn.out_features, "The number of input features of the MLP must be equal to the number of output features of the GNN."
        assert mlp[-1].out_features == num_clusters, "The number of output features of the MLP must be equal to the number of clusters."

        # Hard assignment
        self.hard_assignment = hard_assignment

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # Apply the graph neural network encoder 
        # to obtain the node embeddings
        z = self.gnn(x, adj)
        # Apply the community assignment MLP
        s = self.mlp(z)

        # Pass through softmax to get probabilities
        # for each cluster
        s = F.softmax(s, dim=-1)

        # Hard assignment
        # Convert soft assignment to hard assignment
        if self.hard_assignment:
            s_h = to_hard_assignment(s) 
            return s_h
        else:
            return s
        
### Test CommDetGNN community detection class
if __name__ == '__main__':
    from functions import modularity_loss, v_measure_score
    from graspologic.simulations import sbm
    from graspologic.plot import heatmap
    # modularity = lambda x, y, z: -modularity_loss(x, y, gamma=z)
    ## Create example graph and partition
    # batch, number_of_nodes, number_of_clusters
    b, n_nodes, n_clusters = 1, 100, 3
    
    # Create graphs from a stochastic block model
    n = [n_nodes, n_nodes, n_nodes]
    p = [[0.6, 0.2, 0.3], 
         [0.2, 0.65, 0.2],
         [0.3, 0.2, 0.7]]

    # Print out the graph and community memberships
    print('Graph # of nodes and p values per community:')
    print('n:', n)
    print('p:', p)

    graph, community_memberships = sbm(n=n, p=p, loops=False, return_labels=True)
    # Randomize the memberships order
    community_memberships = torch.tensor(community_memberships).unsqueeze(0)
    community_memberships_p = community_memberships[:,torch.randperm(sum(n))]

    # Convert to one-hot encoding
    partition = torch.functional.F.one_hot(community_memberships_p, num_classes=n_clusters).float()

    # Display the graph
    heatmap(graph.squeeze(), title="Graph w/ 3 Clusters")

    # Graph and node tensors (single graph sample)
    graph = torch.tensor(graph, dtype=torch.float32).unsqueeze(0)
    nodes = torch.eye(sum(n)).unsqueeze(0)

    # Test modularity loss
    print('Initial modularity: {}'.format(-modularity_loss(graph, partition)))
    
    ## Initialize the Community Detection GNN
    # GNN parameters
    in_features = sum(n)
    out_features = 32
    num_layers = 2
    bias = True
    activation = SELU()
    # GNN block
    gnn = GNN(in_features, out_features, num_layers, bias, activation, GCN, batch_norm=True)
    # MLP parameters
    mlp = Sequential(Linear(out_features, n_clusters))
    # Community Detection GNN
    commdet = CommDetGNN(n_clusters, gnn, mlp)

    # Modularity resolution parameter
    gamma = 1.

    ## Train the Community Detection GNN
    # Training parameters
    lr = 0.005
    epochs = 200
    optimizer = torch.optim.Adam(commdet.parameters(), lr=lr)
    patience = 10
    best_loss = float('inf')
    best_partition = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set the model to training mode
    commdet.to(device)
    commdet.train()

    # Train the model
    for epoch in range(epochs):
        # Forward pass
        s = commdet(graph, nodes)
        # Calculate modularity loss
        loss = modularity_loss(graph, s, gamma=gamma)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the best partition
        if loss < best_loss:
            best_loss = loss.item()
            best_partition = s.detach()

        # Early stopping
        if epoch - patience > 0 and loss > best_loss:
            print('Early stopping at epoch: {}'.format(epoch))
            break
        if epoch % 10 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, round(loss.item(),4)))

    # Print the best partition modularity
    print('Best modularity: {}'.format(-best_loss))

    # Evaluate the best partition with actual community memberships
    y = community_memberships
    y_hat = torch.argmax(best_partition, dim=-1)
    # Calculate the accuracy using clustering metric
    # (found clusters may be correct but in different order, thus seemingly incorrect)
    print('V-measure score: {}'.format(v_measure_score(y, y_hat)))

    # Display the best partition
    print(y_hat)


