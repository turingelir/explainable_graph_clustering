import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, v_measure_score, normalized_mutual_info_score
try:
    import community.community_louvain as community_louvain
except ImportError:
    community_louvain = None
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

class CommunityVisualizer:
    """
    A class for visualizing and analyzing community detection results
    """
    def __init__(self, adj_matrix, node_features, communities, ground_truth_labels, community_probs=None):
        """
        Initialize the visualizer with detection results
        
        Args:
            adj_matrix: Adjacency matrix [N, N]
            node_features: Node features [N, F]
            communities: Community assignments [N]
            ground_truth_labels: Ground truth community labels [N] (optional)
            community_probs: Soft community assignments [N, C] (optional)
        """
        self.adj_matrix = adj_matrix.squeeze().cpu().numpy()
        self.node_features = node_features.squeeze().cpu().numpy()
        self.communities = communities.squeeze().cpu().numpy()
        self.ground_truth_labels = ground_truth_labels.squeeze().cpu().numpy()
        self.community_probs = community_probs.squeeze().cpu().numpy() if community_probs is not None else None
        
        # Create NetworkX graph
        self.G = nx.from_numpy_array(self.adj_matrix)
        self.pos = nx.spring_layout(self.G)
        
        # Compute some basic statistics
        self.num_communities = len(np.unique(self.communities))
        self.community_sizes = Counter(self.communities)
        
    def plot_community_graph(self, figsize=(12, 8), node_size=100, save_path=None):
        """Plot the graph with nodes colored by community"""
        # Create figure with a specific layout for colorbar
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique communities and map them to [0, num_communities-1]
        unique_communities = sorted(np.unique(self.communities))
        community_map = {c: i for i, c in enumerate(unique_communities)}
        mapped_communities = np.array([community_map[c] for c in self.communities])
        
        # Create color map
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
        node_colors = colors[mapped_communities]
        
        # Draw the graph
        nx.draw(self.G, pos=self.pos, node_color=node_colors, 
                node_size=node_size, alpha=0.7, with_labels=False,
                ax=ax)
        
        # Add a color bar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, 
                                  norm=plt.Normalize(vmin=0, vmax=len(unique_communities)-1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Community')
        
        plt.title(f'Community Structure Visualization\n({len(unique_communities)} communities)')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_scatter_comparison(self, figsize=(12, 8), save_path=None):
        """
        Create a single scatter plot comparing predicted communities with ground truth labels
        using t-SNE embeddings of node features. Uses different markers for predictions and ground truth.
        """
        if self.ground_truth_labels is None:
            raise ValueError("Ground truth labels are required for comparison visualization")
            
        # Convert labels to integers
        if len(self.ground_truth_labels.shape) > 1:
            ground_truth = np.argmax(self.ground_truth_labels, axis=1)
        else:
            ground_truth = self.ground_truth_labels.astype(int)
        
        communities = self.communities.astype(int)
        
        # Compute t-SNE embedding
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embedding = tsne.fit_transform(self.node_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate correct predictions mask
        correct_predictions = communities == ground_truth
        
        # Plot incorrect predictions first (less emphasis)
        incorrect_mask = ~correct_predictions
        scatter_pred = ax.scatter(embedding[incorrect_mask, 0], embedding[incorrect_mask, 1], 
                                c=communities[incorrect_mask], marker='o', 
                                alpha=0.6, s=100, cmap='tab10',
                                label='Predicted (incorrect)')
        scatter_truth = ax.scatter(embedding[incorrect_mask, 0], embedding[incorrect_mask, 1], 
                                c=ground_truth[incorrect_mask], marker='x', 
                                alpha=0.6, s=100, cmap='tab10',
                                label='Ground Truth')
        
        # Plot correct predictions (more emphasis)
        correct_mask = correct_predictions
        scatter_correct = ax.scatter(embedding[correct_mask, 0], embedding[correct_mask, 1], 
                                c=communities[correct_mask], marker='o',
                                alpha=1.0, s=100, cmap='tab10',
                                label='Correct Predictions')
        
        ax.set_title('Node Embeddings: Predicted vs Ground Truth Communities')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add accuracy information
        accuracy = np.mean(correct_predictions) * 100
        plt.text(0.02, 0.98, f'Accuracy: {accuracy:.1f}%', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, figsize=(10, 8), save_path=None):
        """Plot confusion matrix between predicted and ground truth communities"""
        if self.ground_truth_labels is None:
            raise ValueError("Ground truth labels are required for confusion matrix")
        
        # Convert labels to integers and get unique values
        if len(self.ground_truth_labels.shape) > 1:
            ground_truth = np.argmax(self.ground_truth_labels, axis=1)
        else:
            ground_truth = self.ground_truth_labels.astype(int)
        
        communities = self.communities.astype(int)
        
        # Get unique labels and create mappings
        pred_labels = sorted(np.unique(communities))
        true_labels = sorted(np.unique(ground_truth))
        
        # Create confusion matrix
        conf_matrix = np.zeros((len(true_labels), len(pred_labels)))
        for i in range(len(communities)):
            pred_idx = pred_labels.index(communities[i])
            true_idx = true_labels.index(ground_truth[i])
            conf_matrix[true_idx, pred_idx] += 1
        
        # Normalize by row (ground truth)
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=[f'Pred {i}' for i in pred_labels],
                    yticklabels=[f'True {i}' for i in true_labels])
        
        plt.title('Normalized Confusion Matrix\nGround Truth vs Predicted Communities')
        plt.xlabel('Predicted Community')
        plt.ylabel('Ground Truth Community')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
    def plot_feature_embedding(self, figsize=(10, 10), save_path=None):
        """Plot t-SNE embedding of node features colored by community"""
        # Create figure with a specific layout for colorbar
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique communities and map them to [0, num_communities-1]
        unique_communities = sorted(np.unique(self.communities))
        community_map = {c: i for i, c in enumerate(unique_communities)}
        mapped_communities = np.array([community_map[c] for c in self.communities])
        
        # Compute t-SNE embedding
        tsne = TSNE(n_components=2, random_state=42)
        embedding = tsne.fit_transform(self.node_features)
        
        # Create scatter plot
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=mapped_communities, cmap='rainbow')
        plt.colorbar(scatter, ax=ax, label='Community')
        
        ax.set_title(f't-SNE Visualization of Node Features by Community\n({len(unique_communities)} communities)')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        
    def plot_community_sizes(self, figsize=(10, 6), save_path=None):
        """Plot distribution of community sizes"""
        # Get unique communities and create mapping
        unique_communities = sorted(np.unique(self.communities))
        community_map = {c: i for i, c in enumerate(unique_communities)}
        mapped_communities = np.array([community_map[c] for c in self.communities])
        
        # Count sizes using mapped communities
        sizes = Counter(mapped_communities)
        sorted_sizes = [sizes[i] for i in range(len(unique_communities))]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        bars = ax.bar(range(len(unique_communities)), sorted_sizes)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.xlabel('Community')
        plt.ylabel('Number of Nodes')
        plt.title(f'Community Size Distribution\n({len(unique_communities)} communities)')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        
    def plot_community_connectivity(self, figsize=(8, 8), save_path=None):
        """Plot connectivity matrix between communities"""
        # Get unique communities and create mapping
        unique_communities = sorted(np.unique(self.communities))
        community_map = {c: i for i, c in enumerate(unique_communities)}
        n_communities = len(unique_communities)
        
        # Compute connectivity matrix using mapped indices
        conn_matrix = np.zeros((n_communities, n_communities))
        for i, j in self.G.edges():
            c1, c2 = community_map[self.communities[i]], community_map[self.communities[j]]
            conn_matrix[c1, c2] += 1
            conn_matrix[c2, c1] += 1
            
        # Create figure with a specific layout for colorbar
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(conn_matrix, annot=True, fmt='.0f', cmap='viridis', ax=ax,
                   xticklabels=range(n_communities),
                   yticklabels=range(n_communities))
        
        plt.title(f'Inter-Community Connectivity\n({n_communities} communities)')
        plt.xlabel('Community')
        plt.ylabel('Community')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        
    def compute_metrics(self):
        """Compute various metrics about the community structure"""
        metrics = {}
        
        # Modularity
        try:
            import community.community_louvain as community_louvain
            community_dict = {i: int(self.communities[i]) for i in range(len(self.communities))}
            metrics['modularity'] = community_louvain.modularity(community_dict, self.G)
        except (ImportError, AttributeError):
            # Fallback modularity calculation
            def calculate_modularity(graph, communities):
                m = graph.number_of_edges()
                if m == 0:
                    return 0
                
                Q = 0
                for i, j in graph.edges():
                    if communities[i] == communities[j]:
                        ki = graph.degree(i)
                        kj = graph.degree(j)
                        Q += 1 - (ki * kj) / (2 * m)
                return Q / (2 * m)
            
            metrics['modularity'] = calculate_modularity(self.G, self.communities)
        
        # Silhouette score
        try:
            if self.node_features is not None and len(np.unique(self.communities)) > 1:
                metrics['silhouette'] = silhouette_score(self.node_features, self.communities)
            else:
                metrics['silhouette'] = 0.0
        except:
            metrics['silhouette'] = 0.0
            
        # V-measure and NMI (if ground truth labels are available)
        if self.ground_truth_labels is not None:

            # Convert one-hot encoded labels to class indices if necessary
            if len(self.ground_truth_labels.shape) > 1:
                true_labels = np.argmax(self.ground_truth_labels, axis=1)
            else:
                true_labels = self.ground_truth_labels
            
            print("\nTrue Labels: ", true_labels.shape)
            print("\nPredicted Labels: ", self.communities.shape)
            # Compute similarity metrics

            # V-measure score
            metrics['v_measure'] = v_measure_score(true_labels, self.communities)
            
            # Normalized Mutual Information score
            metrics['nmi'] = normalized_mutual_info_score(
                true_labels, 
                self.communities,
                average_method='geometric'  # Can be changed to 'geometric' or 'min' if needed
            )
        else:
            metrics['v_measure'] = None
            metrics['nmi'] = None
        
                # Conductance for each community
        conductances = []
        print("\nDetailed Conductance Calculation:")
        for comm in np.unique(self.communities):
            try:
                # Get nodes in the community
                comm_nodes = set(np.where(self.communities == comm)[0])
                if len(comm_nodes) == 0:
                    print(f"Community {comm}: Empty community")
                    conductances.append(0.0)
                    continue
                    
                # Calculate internal and external edges
                internal_edges = sum(1 for i, j in self.G.edges() 
                                  if i in comm_nodes and j in comm_nodes)
                external_edges = sum(1 for i, j in self.G.edges() 
                                  if (i in comm_nodes) != (j in comm_nodes))
                
                # Calculate total volume (sum of degrees of nodes in community)
                volume = sum(self.G.degree(i) for i in comm_nodes)
                
                # Print detailed information
                print(f"\nCommunity {comm}:")
                print(f"  Nodes: {len(comm_nodes)}")
                print(f"  Internal edges: {internal_edges}")
                print(f"  External edges: {external_edges}")
                print(f"  Total volume: {volume}")
                
                # Calculate conductance
                if volume == 0:
                    print(f"  Conductance: 0.0 (volume is 0)")
                    conductances.append(0.0)
                else:
                    conductance = external_edges / min(volume, sum(d for n, d in self.G.degree()) - volume)
                    print(f"  Conductance: {conductance:.4f}")
                    conductances.append(conductance)
                    
            except Exception as e:
                print(f"Error in community {comm}: {str(e)}")
                conductances.append(0.0)
                
        metrics['conductance'] = conductances    
        
        """    
        # Conductance for each community
        conductances = []
        for comm in range(self.num_communities):
            try:
                comm_nodes = set(np.where(self.communities == comm)[0])
                other_nodes = set(range(len(self.communities))) - comm_nodes
                cut = sum(1 for i, j in self.G.edges() 
                         if (i in comm_nodes and j in other_nodes) or 
                            (j in comm_nodes and i in other_nodes))
                volume = sum(self.G.degree(i) for i in comm_nodes)
                conductances.append(cut / volume if volume > 0 else 0)
            except:
                conductances.append(0.0)
        metrics['conductance'] = conductances
        """
        return metrics
    
    def plot_interactive_graph(self, save_path=None):
        """Create an interactive plotly visualization of the graph"""
        # Get node positions
        pos = nx.spring_layout(self.G, seed=42)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_x = []
        node_y = []
        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Rainbow',
                color=self.communities,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Community',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Community Structure',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig

def analyze_community_results(data, dataset_name, communities, ground_truth_labels=None, community_probs=None, save_dir=None):
    """
    Comprehensive analysis of community detection results
    
    Args:
        data: Dictionary containing graph data
        communities: Detected community assignments
        community_probs: Soft community assignments (optional)
        save_dir: Directory to save visualizations (optional)
    """
    visualizer = CommunityVisualizer(
        data['adj_matrix'],
        data['node_features'],
        communities,
        ground_truth_labels,
        community_probs
    )
    
    metrics = visualizer.compute_metrics()
    print("\n=== Community Detection Metrics ===")
    print(f"Modularity: {metrics['modularity']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    
    # Print V-measure and NMI if available
    if metrics['v_measure'] is not None:
        print(f"V-measure Score: {metrics['v_measure']:.4f}")
        print(f"Normalized Mutual Information: {metrics['nmi']:.4f}")
    else:
        print("Note: V-measure and NMI not computed (ground truth labels not provided)")
    
    print("\nConductance per community:")
    for i, cond in enumerate(metrics['conductance']):
        print(f"Community {i}: {cond:.4f}")
    
    # Create visualizations
    print("\n=== Generating Visualizations ===")
    
    # Basic community graph
    visualizer.plot_community_graph(
        save_path=f"{save_dir}/{dataset_name}_community_graph.png" if save_dir else None
    )
    
    # Feature embedding
    visualizer.plot_feature_embedding(
        save_path=f"{save_dir}/{dataset_name}_feature_embedding.png" if save_dir else None
    )
    
    # Community sizes
    visualizer.plot_community_sizes(
        save_path=f"{save_dir}/{dataset_name}_community_sizes.png" if save_dir else None
    )
    
    # Community connectivity
    visualizer.plot_community_connectivity(
        save_path=f"{save_dir}/{dataset_name}_community_connectivity.png" if save_dir else None
    )
    
    # Interactive visualization
    fig = visualizer.plot_interactive_graph(
        save_path=f"{save_dir}/{dataset_name}_interactive_graph.html" if save_dir else None
    )
    
    
    # Generate scatter plot comparison if ground truth is available
    if ground_truth_labels is not None:
        print("\n=== Generating Scatter Plot Comparison ===")
        visualizer.plot_scatter_comparison(
            save_path=f"{save_dir}/{dataset_name}_scatter_comparison.png" if save_dir else None
        )
        
        # Confusion matrix
        visualizer.plot_confusion_matrix(
            save_path=f"{save_dir}/{dataset_name}_confusion_matrix.png" if save_dir else None
        )
    
    return visualizer, metrics, fig

"""
# Example usage
if __name__ == '__main__':
    from community_dataloader import get_community_dataloader
    from community_gnn import train_community_detection
    import os
    
    # Create output directory
    os.makedirs('community_viz', exist_ok=True)
    
    # Load and process data
    print("Loading dataset...")
    data = get_community_dataloader('cora')
    
    # Train model
    print("Training model...")
    best_partition, modularity = train_community_detection(data)
    
    # Get community assignments
    communities = torch.argmax(best_partition, dim=-1)
    
    # Analyze and visualize results
    print("Analyzing results...")
    visualizer, metrics, fig = analyze_community_results(
        data,
        communities,
        best_partition,
        save_dir='community_viz'
    )
"""
