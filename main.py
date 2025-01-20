"""
    This is the main file for the project.
    It contains the main method to run the project experiments, given arguments.

    The process is as follows:
        1. Data
             Explanation:
            A list of datasets of graphs are taken as dataloaders: [Citeseer, Amazon, sim(?), ..., etc.]
            Dataloaders are used not to occupy data in memory needlessly and for multiple sample graph datasets generalizability.
             TODO:
            - Create list of dataloaders.
        2. Experiment(s)
            2.1. Community detection methods
                 Explanation:
                Using GNN or iterative greedy algorithm with a target optimization loss (currently modularity and min-cut loss exists).
                There is also clustering regularizer functions ready to prevent degenerate cases.
                ~~~ Future-work: [ignore for now] Use auxilary constraint from ExKMC for optimization target loss.
                 TODO:
                - Create list of obj functions. (2 for now)
                - GNN training, testing method.
                    - Call such method; pass it data, obj func, hyperparam. (curr default param is enough).
                    - Take clustering prediction results.
                    - Save ALL parameters/variables/performance/loss/iteration statistics (you name it) to disk for later eval.
                - Iterative greedy algorithm.
                    - Call such method; pass it data, obj func.
                    - Take clustering prediction results.
                    - Save # iteration statistics, etc. for later eval.
            2.2. ExKMC baseline(s)
                 Explanation:
                ExKMC paper provide 3(?) baselines for evaluation upon [sample x features] data.
                The baselines should be [K-means, Trees, K-means w/ surrogate cost] 
                Using a graph-encoder method (spectral-encoder) node embeddings will be passed to these baselines, 
                and their results will be evaluated upon.
                Also, the case where only node-features (disregarding adj info) are used for clustering will be experimented upon as well.
                 TODO:
                - List of methods and for loop to take their outputs may be created.
                - One loop will only use adjacency graph node embeddings, the other will only use node features
                - Resulting clustering predictions will be returned.
                - Model param (k-means centroids, tree branching info, etc.) must be saved to disk for later eval.
        3. Evaluation & Visualization
            3.1. Data
                - Methods from HFD may be used to display each dataset graph.
                - Datas' graph adjacency matrix using Graspologic Lib. method should be plotted.
                - Datas' graph block-model (connectivity probability between each class) should be plotted using 
                Graspologic Lib. heatmap method. Conn. prob. can be calculated by dividing # edges between each classes
                with # possible edges (non-existing ones included).
                - Datasets using SPECTRAL-ENCODING vectors projected upon a latent space or PCA, should be displayed upon scatter-plots
                with ground-truth class labels.
                - Datasets using NODE FEATURE vectors projected upon a latent space or PCA, should be displayed upon scatter-plots
                with ground-truth class labels.
                ~ [not mandatory] recording each graph's statistics would be phenominal. 
                Node degree distribution, centrality, page-rank statistic etc.
            3.2. Predictions
                - Datasets, using SPECTRAL-ENCODING vectors projected upon a latent space or PCA, should be displayed upon scatter-plots
                with all methods' predictions.
                - Datasets, using NODE FEATURE vectors projected upon a latent space or PCA, should be displayed upon scatter-plots
                with all methods' predictions.
                - Decision trees boundaries should also be displayed using baselines functions.
                Other graph display methods may be used as well. 
            3.3. Performance
                - V-measure, NMI metric table for all methods (comm. det., clustering baselines)
                will be creatd. 
                - Bar-plots may be used.

"""
import os

if 'results' not in os.listdir():
    os.mkdir('results')

import torch
import numpy as np
import pickle

# Import baseline methods
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, v_measure_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from graspologic.plot import heatmap

# Import project modules
from models import SpectralEncoder, ExKMCBaseline
from utils import visualization
from functions import modularity_loss, min_cut_loss, clustering_regularization, generate_graph

from gnn_test import train_community_detection
from datasets import get_community_dataloader

def experiment_GNN(data, obj_func, args):
    r"""
        Method for GNN experiment.
        TODO:
            Add loading ready GNN for only testing
    """
    # Call GNN training method
    if 'fit' in args['modes']:
        res = train_community_detection(data, criterion=obj_func, return_dict=True)
    else: # TODO: Load GNN model and test
        res = {}
    return res

def experiment_kmeans(data, args):
    r"""
        Method for K-means experiment.
        Given samples x features data, apply K-means clustering.
        Output returns clustering predictions, performance metrics and model parameters.
        Args:
            :arg data: Node embeddings or node features data.
            :arg args: Arguments dictionary.
        Returns:
            :return res: Result dictionary.
    """
    # Results dictionary
    res = {}
    # K-means clustering
    kmeans = KMeans(n_clusters=data['num_communities'], random_state=0)
    kmeans.fit(data[data['node_rep']].squeeze())
    # Clustering predictions
    res['prediction'] = kmeans.labels_

    res['model'] = kmeans

    return res

def experiment_exkmc(data, args):
    r"""
        Method for ExKMC experiment.
        Given samples x features data, apply ExKMC clustering.
        Output returns clustering predictions, performance metrics and model parameters.
        Args:
            :arg data: Node embeddings or node features data.
            :arg args: Arguments dictionary.
        Returns:
            :return res: Result dictionary.
    """
    # Results dictionary
    res = {}
    # ExKMC clustering
    baseline = ExKMCBaseline(num_clusters=data['num_communities'], num_components=data['num_communities'], random_state=0)
    # Visualize tree
    if 'visualize' in args['modes']:
        baseline.fit_and_plot_exkmc(data[data['node_rep']], title= data['dataset_name'] + data['node_rep'] + ' ExKMC')
    else:
        baseline.fit(data[data['node_rep']].squeeze())
    # Clustering predictions
    res['prediction'] = baseline.labels_

    res['model'] = baseline

    return res


def experiment(data, method_name, args):
    r"""
        This method is the main general experiment method for the project.
        Given a graph sample data, a method is applied and results are evaluated.
        Each method returns a clustering prediction result, performance metrics and model parameters.
        These results are saved to disk for later evaluation.
        According to visualization and evaluation modes, the results are displayed.
        Args:
            :arg data: Graph dict data or node embeddings/features data.
            :arg method_name: Method name to apply on data.
            :arg args: Arguments dictionary.
    """
    # Results dictionary
    res = {}

    # Pass data to appropriate method experiment.
    if method_name == 'GNN':
        # Call GNN method
        res = experiment_GNN(data, args['obj_funcs'][0], args)
    elif method_name == 'IterativeGreedy':
        # Call IterativeGreedy method
        pass
    elif method_name == 'K-means':
        # Call K-means method
        res = experiment_kmeans(data, args)
    elif method_name == 'IMM':
        # Call Trees method
        pass
    elif method_name == 'ExKMC':
        # Call ExKMC method
        res = experiment_exkmc(data, args)
    else:
        raise NotImplementedError(f"Method {method_name} is not implemented.")
    
    return res

def save_results(results, path):
    r"""
        Save results dictionary elements to disk.
        Create seperate folders for each dataset and method.
        Save results as pickle files for later evaluation.
        Args:
            :arg results: Results dictionary.
                Where key is (dataset_name, method_name) tuple.
                and value is the result dictionary.
            :arg args: Arguments dictionary.
    """
    # Create folders for each dataset and method
    for (dataset_name, method_name), res in results.items():
        folder_path = os.path.join(path, dataset_name, method_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # Iterate over result dictionary and save each element
        for key, val in res.items():
            # Save results to disk
            # Check val data type and save accordingly
            # Torch tensor
            if isinstance(val, torch.Tensor) or isinstance(val, torch.nn.Module):
                torch.save(val, os.path.join(folder_path, f"{method_name}_{key}.pt"))
            # Numpy array
            elif isinstance(val, np.ndarray):
                np.save(os.path.join(folder_path, f"{method_name}_{key}.npy"), val)
            # Class object
            elif isinstance(val, object):
                with open(os.path.join(folder_path, f"{method_name}_{key}.pkl"), 'wb') as f:
                    pickle.dump(val, f)
            else:
                with open(os.path.join(folder_path, f"{method_name}_{key}.txt"), 'w') as f:
                    f.write(val)

def main(args):
    ####        1. Data        ####
    # Create list of dataloaders.
    # dataloaders = [Citeseer, Amazon, sim(?), ..., etc.]
    # TODO: SBM generated graph for now
    graph = generate_graph()
    
    # Datasets list 
    datasets = ['sim']

    ####        2. Experiment(s)        ####
    # Results dictionary
    results = {}
    # Call experiment method over each dataset and method 
    for dataset_name in datasets:
        # Load data
        if dataset_name == 'sim':
            # Create graph data
            data = graph
        else:
            data = get_community_dataloader(dataset_name)
            # Add dataset name to data
            data['dataset_name'] = dataset_name

        # Baseline methods only work on sample x features data
        # So, we need to extract node embeddings or node features from data.
        # We can use SpectralEncoder for this purpose.
        # If we are using baselines, encode graph data
        if len(args['baselines']) > 0:
            encoder = SpectralEncoder(data['num_communities'], norm_laplacian=True)
            # Encode graph data
            data['node_embeddings'] = encoder.fit_transform(data['adj_matrix'].squeeze()).unsqueeze(0)
        
        # Do experiment for each method
        # Seperate community detection methods and baselines
        comm_det_methods = [method_name for method_name in args['methods'] if method_name not in args['baselines']]
        baselines = [method_name for method_name in args['methods'] if method_name in args['baselines']]
        # Community detection methods
        for method_name in comm_det_methods:
            results[(dataset_name, method_name)] = experiment(data, method_name, args)
        # Baseline clustering methods
        for method_name in baselines:
            # Do experiment for each node representation type: embeddings, features
            for node_rep in args['node_rep']:
                # Give what node representation to use
                data['node_rep'] = node_rep
                results[(dataset_name, method_name + '_' + node_rep)] = experiment(data, method_name, args)

    # Save results to disk
    save_results(results, args['save_path'])

    ####        3. Evaluation & Visualization        ####
    # Visualize graphs
    if 'graphs' in args['visualize']:
        for dataset_name, data in datasets.items():
            graph_adj = data['adj_matrix']
            # Graph adjacency matrix
            visualization.show_mat(graph_adj.squeeze(), dataset_name, show=args['show'], save=args['save'], 
                                   save_path=os.path.join(args['save_path'], dataset_name), cmap='binary')
            # Graph block-model
            
            


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(0)

    # Take arguments
    args = {'modes': ['fit', 'eval', 'visualize', ], # 'load', 
            'methods': ['GNN', 'K-means', 'ExKMC'], # 'IterativeGreedy', 'IMM',
            'baselines': ['K-means', 'ExKMC'], # 'IMM', 
            'node_rep': ['node_embeddings', 'node_features'],
            'datasets': ['sim'], # 'Citeseer', 'Amazon', 
            'obj_funcs': [modularity_loss], # 'min-cut' 
            'visualize': ['graphs', 'predictions', 'performance'],
            'eval': ['V-measure', 'NMI'],
            'dim_red': ['PCA'], # 't-SNE'
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_path': os.path.join(os.getcwd(), 'results'),
            'show': False,
            'save': True
            }
    assert not(args['modes'].count('fit') and args['modes'].count('load')), "Only fit or load mode can be selected at a time."

    # Call main method
    main(args)