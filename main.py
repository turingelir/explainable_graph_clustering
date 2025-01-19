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

import torch

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(0)

    # Take arguments
    