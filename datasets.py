import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

class CommunityDataLoader:
    """
    Dataloader class specifically designed for community detection tasks.
    Handles various graph datasets and prepares them for community detection.
    """
    def __init__(self, dataset_name, root='./data',  num_communities=None):
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.transform = NormalizeFeatures()
        self.num_communities = num_communities
        
    def estimate_communities(self, adj_matrix):
        """Estimate number of communities using spectral clustering if not provided"""
        # Convert to networkx graph for analysis
        G = nx.from_scipy_sparse_matrix(adj_matrix) if hasattr(adj_matrix, 'todense') else nx.from_numpy_array(adj_matrix)
        
        # Use modularity-based community detection
        try:
            from community import community_louvain
            communities = community_louvain.best_partition(G)
            return len(set(communities.values()))
        except:
            # Fallback to degree-based estimation
            degrees = [d for n, d in G.degree()]
            return min(len(G) // 20, int(np.sqrt(len(G))))  # Heuristic estimation

    def prepare_for_community_detection(self, data):
        """Prepare data for community detection tasks"""
        # Convert to dense adjacency matrix if needed
        if hasattr(data, 'edge_index'):
            adj = torch.zeros((data.num_nodes, data.num_nodes))
            adj[data.edge_index[0], data.edge_index[1]] = 1
        else:
            adj = data
            
        # Estimate number of communities if not provided
        if self.num_communities is None:
            self.num_communities = self.estimate_communities(adj.numpy())
            
        # Create initial community assignments using degree-based clustering
        degrees = adj.sum(dim=1)
        kmeans = KMeans(n_clusters=self.num_communities, random_state=42)
        initial_communities = kmeans.fit_predict(degrees.reshape(-1, 1))
        
        # Convert to one-hot encoding
        community_assignments = torch.zeros((adj.shape[0], self.num_communities))
        community_assignments[torch.arange(adj.shape[0]), initial_communities] = 1
        
        # Create node features if not present
        if not hasattr(data, 'x') or data.x is None:
            node_features = torch.eye(adj.shape[0])  # Use one-hot encoding as features
        else:
            node_features = data.x
            
        return {
            'adj_matrix': adj.unsqueeze(0),  # Add batch dimension
            'node_features': node_features.unsqueeze(0),  # Add batch dimension
            'initial_communities': community_assignments.unsqueeze(0),  # Add batch dimension
            'num_communities': self.num_communities
        }

    def load_data(self):
        """Load and prepare the specified dataset for community detection"""
        # Handle OGB-arXiv dataset
        if self.dataset_name in ['ogb-arxiv', 'arxiv', 'ogbarxiv']:
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
            data = dataset[0]
            
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

def get_community_dataloader(dataset_name, num_communities=None):
    """Utility function to easily load datasets for community detection"""
    loader = CommunityDataLoader(dataset_name=dataset_name, num_communities=num_communities)
    return loader.load_data()