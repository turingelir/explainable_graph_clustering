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
from datasets import get_community_dataloader

from functions import to_hard_assignment
from functions import modularity_loss
from functions.metric import v_measure_score
from comdet_visualizer import analyze_community_results
from community_logger import CommunityLogger

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
    """ Modified Graph Neural Network (GNN) community detection module."""
    def __init__(self, num_clusters: int, gnn: Module, mlp: Module,
                 hard_assignment: bool = False):
        super(CommDetGNN, self).__init__()
        # Number of target clusters
        self.num_clusters = num_clusters
        # Graph neural network encoder parameters
        self.gnn = gnn
        # Community assignment MLP
        self.mlp = mlp
        assert mlp[-1].out_features == num_clusters, "MLP output must match number of clusters"

        # Hard assignment
        self.hard_assignment = hard_assignment

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Apply the graph neural network encoder 
        z = self.gnn(x, adj)
        # Apply the community assignment MLP
        s = self.mlp(z)
        # Pass through softmax to get probabilities
        s = F.softmax(s, dim=-1)
        
        # Hard assignment if specified
        if self.hard_assignment:
            return to_hard_assignment(s)
        return s

def train_community_detection(data, device='cuda' if torch.cuda.is_available() else 'cpu',
                            lr=0.005, epochs=200, patience=10, 
                            criterion=modularity_loss, return_dict=False):
    """
    Train the community detection model on the provided dataset.
    
    Args:
        data: Dictionary containing:
            - adj_matrix: Adjacency matrix [B, N, N]
            - node_features: Node features [B, N, F]
            - initial_communities: Initial community assignments [B, N, C]
            - num_communities: Number of communities
        device: Device to train on
        lr: Learning rate
        epochs: Maximum number of epochs
        patience: Early stopping patience
    """
    # Get dimensions from data
    num_nodes = data['node_features'].shape[1]
    num_features = data['node_features'].shape[2]
    num_clusters = data['num_communities']

    # Initialize model components
    gnn = GNN(
        in_features=num_features,
        out_features=32,  # Hidden dimension
        num_layers=2,
        bias=True,
        activation=SELU(),
        batch_norm=True
    )
    
    mlp = Sequential(
        Linear(32, num_clusters)
    )
    
    # Initialize the community detection model
    model = CommDetGNN(num_clusters, gnn, mlp)
    model = model.to(device)
    
    # Move data to device
    adj_matrix = data['adj_matrix'].to(device)
    node_features = data['node_features'].to(device)
    initial_communities = data['initial_communities'].to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_loss = float('inf')
    best_partition = None
    no_improve = 0
    
    model.train()
    for epoch in range(epochs):
        # Forward pass
        communities = model(node_features, adj_matrix)
        
        # Calculate loss (negative modularity as we want to maximize modularity)
        loss = criterion(adj_matrix, communities)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track best performance
        if loss < best_loss:
            best_loss = loss.item()
            best_partition = communities.detach()
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    # Save results to dict
    res = {
        'model': model.state_dict(),
        'prediction': best_partition,
        'best_loss': -best_loss # Return positive modularity
    }
    if return_dict:
        return res
    return best_partition, -best_loss  # Return positive modularity
        
if __name__ == '__main__':
    



    # Initialize dataset and logger
    dataset_name = 'citeseer'
    logger = CommunityLogger(dataset_name)
    
    try:
        # Log dataset loading
        logger.log_info(f"Loading {dataset_name} dataset...")
        data = get_community_dataloader(dataset_name)
        
        # Log dataset information
        data_info = {
            'num_nodes': data['node_features'].shape[1],
            'num_features': data['node_features'].shape[2],
            'num_communities': data['num_communities']
        }
        logger.log_dataset_info(data_info)
        
        # Train model
        best_partition, modularity = train_community_detection(data)
        communities = torch.argmax(best_partition, dim=-1)

        ground_truth_labels = data['initial_communities']
        
        # Log shapes for debugging
        debug_info = {
            'Predicted communities shape': str(communities.shape),
            'Ground truth labels shape': str(ground_truth_labels.shape),
            'Unique predicted communities': len(torch.unique(communities)),
            'Unique ground truth communities': len(torch.unique(torch.argmax(ground_truth_labels, dim=-1)))
        }
        logger.log_metrics(debug_info, "Debug Information")

        # Analyze results
        visualizer, metrics, fig = analyze_community_results(
            data,
            dataset_name,
            communities,
            ground_truth_labels,
            best_partition,
            save_dir='community_viz'
        )

        # Log final metrics
        logger.log_metrics(metrics, "Clustering Evaluation Metrics")
        
    except Exception as e:
        logger.log_error(str(e))
        raise e   
   
    
    """
    # Load and process data
    dataset_name = 'citeseer'
    data = get_community_dataloader(dataset_name)
    print(f"Loading {dataset_name} dataset...")
    

    # Train model
    best_partition, modularity = train_community_detection(data)
    communities = torch.argmax(best_partition, dim=-1)

    ground_truth_labels = data['initial_communities']  # This should contain the ground truth communities

    # Print shapes for debugging
    print(f"\nDebug Information:")
    print(f"Predicted communities shape: {communities.shape}")
    print(f"Ground truth labels shape: {ground_truth_labels.shape}")
    print(f"Number of unique predicted communities: {len(torch.unique(communities))}")
    print(f"Number of unique ground truth communities: {len(torch.unique(torch.argmax(ground_truth_labels, dim=-1)))}")

    visualizer, metrics, fig = analyze_community_results(
        data,
        dataset_name,
        communities,
        ground_truth_labels,
        best_partition,
        save_dir='community_viz'
    )

    # Print the metrics
    print("\nClustering Evaluation Metrics:")
    print(f"V-measure: {metrics['v_measure']:.4f}")
    print(f"NMI Score: {metrics['nmi']:.4f}")
    print(f"Modularity: {metrics['modularity']:.4f}")

    #visualizer.plot_community_graph()
    #visualizer.plot_feature_embedding()
    #visualizer.plot_community_sizes()
    #visualizer.plot_community_connectivity()
    
"""


