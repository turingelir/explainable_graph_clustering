import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx
import numpy as np

class CommunityDataLoader:
    """
    Dataloader class specifically designed for community detection tasks.
    Uses actual dataset labels instead of KMeans clustering.
    """
    def __init__(self, dataset_name, root='./data'):
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.transform = NormalizeFeatures()
        
    def prepare_for_community_detection(self, data):
        """Prepare data for community detection tasks using actual labels"""
        # Convert to dense adjacency matrix if needed
        if hasattr(data, 'edge_index'):
            adj = torch.zeros((data.num_nodes, data.num_nodes))
            adj[data.edge_index[0], data.edge_index[1]] = 1
        else:
            adj = data
            
        # Get actual labels from the dataset
        if hasattr(data, 'y'):
            labels = data.y
        else:
            raise ValueError("Dataset does not contain labels (y attribute)")
            
        # Convert labels to zero-based continuous indexing if needed
        unique_labels = torch.unique(labels)
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        mapped_labels = torch.tensor([label_map[label.item()] for label in labels])
        
        num_communities = len(unique_labels)
        
        # Convert to one-hot encoding
        community_assignments = torch.zeros((adj.shape[0], num_communities))
        community_assignments[torch.arange(adj.shape[0]), mapped_labels] = 1
        
        # Create node features if not present
        if not hasattr(data, 'x') or data.x is None:
            node_features = torch.eye(adj.shape[0])  # Use one-hot encoding as features
        else:
            node_features = data.x
            
        return {
            'adj_matrix': adj.unsqueeze(0),  # Add batch dimension
            'node_features': node_features.unsqueeze(0),  # Add batch dimension
            'initial_communities': community_assignments.unsqueeze(0),  # Add batch dimension
            'num_communities': num_communities,
            'true_labels': mapped_labels  # Also return the original labels for evaluation
        }

    def load_data(self):
        """Load and prepare the specified dataset for community detection"""
        # Handle OGB-arXiv dataset
        if self.dataset_name in ['ogb-arxiv', 'arxiv', 'ogbarxiv']:
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
            data = dataset[0]
            # For OGB datasets, labels are stored differently
            data.y = data.y.squeeze()
            
        # Handle Planetoid datasets
        elif self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(
                root=self.root,
                name=self.dataset_name.capitalize(),
                transform=self.transform
            )
            data = dataset[0]
            
        # Handle Amazon datasets
        elif 'amazon' in self.dataset_name:
            name = 'Computers' if 'computer' in self.dataset_name else 'Photo'
            dataset = Amazon(root=self.root, name=name, transform=self.transform)
            data = dataset[0]
            
        # Handle Coauthor datasets
        elif 'coauthor' in self.dataset_name:
            name = 'CS' if 'cs' in self.dataset_name else 'Physics'
            dataset = Coauthor(root=self.root, name=name, transform=self.transform)
            data = dataset[0]
            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
            
        return self.prepare_for_community_detection(data)

def get_community_dataloader(dataset_name):
    """Utility function to easily load datasets for community detection"""
    loader = CommunityDataLoader(dataset_name=dataset_name)
    return loader.load_data()
