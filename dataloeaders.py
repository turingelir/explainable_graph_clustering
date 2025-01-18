import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from sklearn.datasets import load_iris, load_wine, load_digits
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
import urllib.request
import scipy.sparse as sp
from tqdm import tqdm
import zipfile
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset


class ExtendedGraphDatasetLoader:
    """
    Extended loader class that includes PyTorch Geometric datasets, 
    custom coauthor datasets, and OGB datasets
    """
    def __init__(self, root='./data', dataset_name='Cora'):
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.transform = NormalizeFeatures()
        
    def load_data(self):
        """Load and return the specified dataset"""
        
        # Handle OGB-arXiv dataset
        if self.dataset_name in ['ogb-arxiv', 'arxiv', 'ogbarxiv']:
            try:
                dataset = PygNodePropPredDataset(
                    name='ogbn-arxiv',
                    root=self.root
                )
                data = dataset[0]  # Get the first graph
                
                # Add some useful properties
                data.num_classes = dataset.num_classes
                data.num_features = dataset.num_features
                
                return data
                
            except Exception as e:
                raise ValueError(
                    f"Error loading OGB-arXiv dataset. Make sure you have installed "
                    f"the 'ogb' package using: pip install ogb\nError: {str(e)}"
                )
        
        # Handle Planetoid datasets (Cora, Citeseer, PubMed)
        elif self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(
                root=self.root,
                name=self.dataset_name.capitalize(),
                transform=self.transform
            )
        
        # Handle Amazon datasets
        elif 'amazon' in self.dataset_name:
            amazon_map = {
                'amazon': 'Computers',
                'amazonpc': 'Computers',
                'amazoncomputers': 'Computers',
                'amazonphoto': 'Photo',
                'amazonphotos': 'Photo'
            }
            
            name = amazon_map.get(self.dataset_name, 'Computers')
            dataset = Amazon(
                root=self.root,
                name=name,
                transform=self.transform
            )
            
        # Handle Coauthor datasets
        elif 'coauthor' in self.dataset_name:
            coauthor_map = {
                'coauthor': 'CS',
                'coauthorcs': 'CS',
                'coauthorphys': 'Physics',
                'coauthorphysics': 'Physics'
            }
            
            if self.dataset_name not in coauthor_map:
                raise ValueError(
                    f"Invalid Coauthor dataset. Only 'CoauthorCS' and 'CoauthorPhysics' are "
                    f"supported in PyTorch Geometric. For other coauthor datasets, "
                    f"use the CustomCoauthorDataset class."
                )
            
            name = coauthor_map[self.dataset_name]
            dataset = Coauthor(
                root=self.root,
                name=name,
                transform=self.transform
            )
            
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported.\n"
                f"Supported datasets:\n"
                f"- Citation networks: Cora, Citeseer, PubMed\n"
                f"- Amazon co-purchase: AmazonPC/Computers, AmazonPhoto\n"
                f"- Coauthor networks: CoauthorCS, CoauthorPhysics\n"
                f"- Paper citation: OGB-arXiv"
            )
            
        return dataset
    

class TraditionalDataset(Dataset):
    """
    Dataset class for traditional ML datasets (Iris, Wine, Digits)
    """
    def __init__(self, dataset_name='iris', train=True):
        self.dataset_name = dataset_name.lower()
        
        # Load the appropriate dataset
        if self.dataset_name == 'iris':
            data = load_iris()
        elif self.dataset_name == 'wine':
            data = load_wine()
        elif self.dataset_name == 'digits':
            data = load_digits()
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
            
        self.features = torch.FloatTensor(data.data)
        self.targets = torch.LongTensor(data.target)
        
        # Split into train/test (80/20 split)
        n_samples = len(self.features)
        indices = torch.randperm(n_samples)
        split = int(0.8 * n_samples)
        
        if train:
            self.features = self.features[indices[:split]]
            self.targets = self.targets[indices[:split]]
        else:
            self.features = self.features[indices[split:]]
            self.targets = self.targets[indices[split:]]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
class CustomCoauthorDataset(InMemoryDataset):
    """
    Custom dataset class for loading additional coauthor datasets
    (Medicine, Chemistry, Engineering) from the GNN benchmark repository
    """
    
    url_base = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"
    datasets = {
        'medicine': 'ms_academic_med.npz',
        'chemistry': 'ms_academic_chem.npz',
        'engineering': 'ms_academic_eng.npz'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        if self.name not in self.datasets:
            raise ValueError(f"Dataset {name} not found. Available datasets: {list(self.datasets.keys())}")
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.datasets[self.name]]

    @property
    def processed_file_names(self):
        return [f'coauthor_{self.name}_data.pt']

    def download(self):
        # Download the file from the GNN benchmark repository
        url = self.url_base + self.datasets[self.name]
        filename = os.path.join(self.raw_dir, self.datasets[self.name])
        
        if not os.path.exists(filename):
            print(f'Downloading {self.name} dataset...')
            os.makedirs(self.raw_dir, exist_ok=True)
            urllib.request.urlretrieve(url, filename)

    def process(self):
        # Load the npz file
        data = np.load(os.path.join(self.raw_dir, self.datasets[self.name]))
        
        # Extract adjacency matrix and features
        adj_matrix = sp.csr_matrix((data['adj_data'], data['adj_indices'], 
                                  data['adj_indptr']), shape=data['adj_shape'])
        
        # Convert to edge_index format
        edge_index = torch.tensor(np.vstack(adj_matrix.nonzero()), dtype=torch.long)
        
        # Process features
        features = torch.tensor(data['features'].todense() if sp.issparse(data['features']) 
                              else data['features'], dtype=torch.float)
        
        # Process labels
        labels = torch.tensor(data['labels'] if 'labels' in data 
                            else np.zeros(features.shape[0]), dtype=torch.long)

        # Create Data object
        data = Data(x=features, 
                   edge_index=edge_index, 
                   y=labels)

        # Save processed data
        torch.save(self.collate([data]), self.processed_paths[0])


def get_dataloader(dataset_name, batch_size=32, is_graph=True, train=True):
    """
    Updated utility function that uses the ExtendedGraphDatasetLoader
    """
    if is_graph:
        dataset_loader = ExtendedGraphDatasetLoader(dataset_name=dataset_name)
        return dataset_loader.load_data()
    else:
        # For traditional datasets
        if dataset_name.lower() == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = CIFAR10(root='./data', train=train, download=True, transform=transform)
        else:
            dataset = TraditionalDataset(dataset_name, train=train)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=2
        )

"""
# Load a graph dataset
graph_data = get_dataloader('Cora', is_graph=True)

# Access the data
x = graph_data.data.x  # Node features
edge_index = graph_data.data.edge_index  # Graph connectivity
y = graph_data.data.y  # Labels
"""


def inspect_graph_dataset(dataset_name):
    """Inspect and print information about a graph dataset"""
    print(f"\n=== Inspecting {dataset_name} Dataset ===")
    dataset = get_dataloader(dataset_name, is_graph=True)
    data = dataset[0]  # Get the first graph
    
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Number of node features: {data.x.shape[1]}")
    print(f"Number of classes: {len(torch.unique(data.y))}")
    print("\nFeature matrix shape:", data.x.shape)
    print("First few features of first node:", data.x[0][:5])
    print("\nFirst few edges:", data.edge_index[:, :5])
    
    # Visualize a small subgraph
    G = nx.Graph()
    edges = data.edge_index.t().numpy()
    G.add_edges_from(edges[:100])  # Add first 100 edges
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, node_size=50, node_color='blue', alpha=0.6)
    plt.title(f"Subgraph visualization of {dataset_name}")
    plt.show()


def inspect_traditional_dataset(dataset_name):
    """Inspect and print information about a traditional dataset"""
    print(f"\n=== Inspecting {dataset_name} Dataset ===")
    train_loader = get_dataloader(dataset_name, batch_size=32, is_graph=False, train=True)
    test_loader = get_dataloader(dataset_name, batch_size=32, is_graph=False, train=False)
    
    # Get a batch of data
    features, labels = next(iter(train_loader))
    
    print("Training set size:", len(train_loader.dataset))
    print("Test set size:", len(test_loader.dataset))
    print("Feature shape:", features.shape)
    print("Number of classes:", len(torch.unique(labels)))
    print("\nFirst batch shape:", features.shape)
    print("First few features of first sample:", features[0][:5])
    print("First few labels:", labels[:5])
    
    # For 2D visualization of first two features
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
    plt.title(f"First two features of {dataset_name}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Class')
    plt.show()



# Test function
def test_dataloaders():
    # Test graph datasets
    print("\nTesting Graph Datasets:")
    graph_datasets = ['Cora', 'Citeseer','PubMed', 'Amazon', 'Coauthor', 'ogb-arxiv']
    for dataset_name in graph_datasets:
        try:
            inspect_graph_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
    
    # Test traditional datasets
    print("\nTesting Traditional Datasets:")
    traditional_datasets = ['iris', 'wine', 'digits']
    for dataset_name in traditional_datasets:
        try:
            inspect_traditional_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")


"""

#Example usage for datasets
# Load a graph dataset (e.g., Cora)
graph_data = get_dataloader('Cora', is_graph=True)

# Access the data
data = graph_data[0]  # Get the first graph
x = data.x  # Node features
edge_index = data.edge_index  # Graph connectivity
y = data.y  # Labels

print(f"Feature matrix shape: {x.shape}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Labels shape: {y.shape}")


# Load training data
train_loader = get_dataloader('iris', batch_size=32, is_graph=False, train=True)
# Load test data
test_loader = get_dataloader('iris', batch_size=32, is_graph=False, train=False)

# Example: iterate through batches
for batch_idx, (features, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    break  # Just showing the first batch

# Load CIFAR-10
train_loader = get_dataloader('cifar10', batch_size=32, is_graph=False, train=True)
test_loader = get_dataloader('cifar10', batch_size=32, is_graph=False, train=False)

# Check a batch
images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")  # Should be [32, 3, 32, 32]
print(f"Labels shape: {labels.shape}")       # Should be [32]

"""

"""
cora = get_dataloader('cora')
citeseer = get_dataloader('citeseer')
pubmed = get_dataloader('pubmed')

amazon_pc = get_dataloader('amazonpc')
amazon_photo = get_dataloader('amazonphoto')

coauthor_cs = get_dataloader('coauthorcs')
coauthor_physics = get_dataloader('coauthorphys')


# Load the OGB-arXiv dataset
arxiv_data = get_dataloader('ogb-arxiv')

# Access the data
x = arxiv_data.x  # Node features
edge_index = arxiv_data.edge_index  # Graph connectivity
y = arxiv_data.y  # Labels

# Print dataset statistics
print(f"Number of nodes: {x.shape[0]}")
print(f"Number of features: {x.shape[1]}")
print(f"Number of edges: {edge_index.shape[1] // 2}")
print(f"Number of classes: {arxiv_data.num_classes}")


#Burada bir sıkıntı var.

# Load the additional coauthor datasets
coauthor_med = get_dataloader('coauthormed')
coauthor_chem = get_dataloader('coauthorchem')
coauthor_eng = get_dataloader('coauthoreng')

# Example: Print dataset statistics
print(f"Medicine dataset:")
print(f"Number of nodes: {coauthor_med.data.x.shape[0]}")
print(f"Number of features: {coauthor_med.data.x.shape[1]}")
print(f"Number of edges: {coauthor_med.data.edge_index.shape[1] // 2}")
"""