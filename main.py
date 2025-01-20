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

from models import SpectralEncoder
from utils import visualization
from functions import modularity_loss, min_cut_loss, clustering_regularizer, generate_graph

from gnn_test import train_community_detection

def exp_GNN(data, obj_func, args):
    r"""
        Method for GNN experiment.
        TODO:
            Add loading ready GNN for only testing
    """
    # Call GNN training method
    if 'train' in args['modes']:
        res = train_community_detection(data, criterion=obj_func, return_dict=True)
    else:
        res = {}
    return res

def experiment(data, method_name, args):
    r"""
        This method is the main general experiment method for the project.
        Given a graph sample data, a method is applied and results are evaluated.
        Each method returns a clustering prediction result, performance metrics and model parameters.
        These results are saved to disk for later evaluation.
        According to visualization and evaluation modes, the results are displayed.
        Args:
            :arg data: Graph data sample.
            :arg method_name: Method name to apply on data.
            :arg args: Arguments dictionary.
    """
    # Pass data to appropriate method experiment.
    if method_name == 'GNN':
        # Call GNN method
        res = exp_GNN(data, args['obj_funcs'][0], args)
    elif method_name == 'IterativeGreedy':
        # Call IterativeGreedy method
        pass
    elif method_name == 'K-means':
        # Call K-means method
        pass
    elif method_name == 'Trees':
        # Call Trees method
        pass
    elif method_name == 'ExKMC':
        # Call ExKMC method
        pass
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
            if isinstance(val, torch.Tensor):
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
    datasets = {'sim': graph}

    ####        2. Experiment(s)        ####
    # Results dictionary
    results = {}
    # Call experiment method over each dataset and method 
    for dataset_name, data in datasets.items():
        for method_name in args['methods']:
            results[(dataset_name, method_name)] = experiment(data, method_name, args)
    # Save results to disk
    save_results(results, args['save_path'])
                


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(0)

    # Take arguments
    args = {'modes': ['eval', 'visualize'], # 'train', 
            'methods': ['GNN', 'IterativeGreedy', 'K-means', 'Trees', 'ExKMC'], 
            'datasets': ['sim'], # 'Citeseer', 'Amazon', 
            'obj_funcs': [modularity_loss], # 'min-cut' 
            'visualize': ['graphs', 'predictions', 'performance'],
            'eval': ['V-measure', 'NMI'],
            'dim_red': ['PCA'], # 't-SNE'
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_path': os.path.join(os.getcwd(), 'results'),
            }
    # Call main method
    main(args)