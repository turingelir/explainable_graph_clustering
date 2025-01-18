"""
    This module contains the spectral eigenvector encoding method for graph clustering.
    For a given graph's adjacency matrix, the spectral method computes the eigenvectors of the Laplacian matrix.
    The eigenvectors that correspond to the smallest eigenvalues are used to encode the graph.
"""

import torch

class SpectralEncoder:
    def __init__(self, n_clusters=3, *, eigen_solver='eigh', n_components=None,
                    random_state=None):
        r"""
            Spectral clustering algorithm, eigenvector encoding method.
            This algorithm computes the eigenvectors of the Laplacian matrix of the graph.
            The eigenvectors that correspond to the smallest eigenvalues are used to encode the graph.

            Parameters:
            -----------
                n_clusters : int, default=8
                    The number of clusters to form as well as the number of centroids to generate.

                eigen_solver : {'eigh'}, default='eigh'
                    The eigenvalue decomposition strategy to use.
                    'eigh' uses the eig function from torch.linalg.eigh.
                        Affinity matrix is assumed to be symmetric.

                n_components : int, default=None
                    Number of eigenvectors to use for the spectral embedding.
                    If None, use n_clusters.

                random_state : int, RandomState instance or None, default=None
                    A pseudo random number generator used for the initialization of the lobpcg eigenvectors
                    decomposition when eigen_solver == 'amg'.

            Attributes:
            -----------
                affinity_matrix_ : array-like of shape (n_samples, n_samples)
                    Affinity matrix used for clustering. This is the adjacency matrix of the graph.

                eigenvalues_ : array-like of shape (n_components,)
                    The eigenvalues of the affinity matrix.

                eigenvectors_ : array-like of shape (n_samples, n_components)
                    The eigenvectors of the affinity matrix.

                labels_ : array-like of shape (n_samples,)
                    Labels (clusters) of the input data.
        """
        # Parameters
        self.n_clusters = n_clusters
        if eigen_solver not in ['eigh']:
            raise NotImplementedError(f"eigen_solver={eigen_solver} is not implemented. Use 'eigh'.")
        else:
            self.eigen_solver = torch.linalg.eigh
        if n_components is not None:
            assert n_components <= n_clusters, "n_components must be less than or equal to n_clusters."
            self.n_components = n_components
        else:
            self.n_components = n_clusters
        self.random_state = random_state

        # Attributes
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def __laplacian(self, A):
        r"""
            Compute the Laplacian matrix of the graph.

            Parameters:
            -----------
                A : array-like of shape (n_samples, n_samples)
                    The adjacency matrix of the graph.

            Returns:
            --------
                L : array-like of shape (n_samples, n_samples)
                    The Laplacian matrix of the graph.
        """
        # Compute the degree matrix
        D = torch.diag(torch.sum(A, dim=-1))

        # Compute the Laplacian matrix
        L = D - A

        return L

    def fit(self, A):
        r"""
            Compute the spectral eigenvectors of the Laplacian matrix of the graph.

            Parameters:
            -----------
                A : array-like of shape (n_samples, n_samples)
                    The adjacency matrix of the graph.

            Returns:
            --------
                self : object
                    Returns the instance itself.
        """
        # Compute the Laplacian matrix
        L = self.__laplacian(A)

        # Compute the eigenvectors of the Laplacian matrix
        # eigenvalues are in ascending order
        eigenvalues, eigenvectors = self.eigen_solver(L)

        # Select the eigenvectors that correspond to the smallest n_components eigenvalues
        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.eigenvectors_ = eigenvectors[:, :self.n_components]

        return self
    
    def get_embeddings(self):
        r"""
            Get the spectral embedding of the graph.

            Returns:
            --------
                embedding : array-like of shape (n_samples, n_components)
                    The spectral embedding of the graph.
        """
        return self.eigenvectors_


