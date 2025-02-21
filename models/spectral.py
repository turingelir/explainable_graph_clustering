"""
    This module contains the spectral eigenvector encoding method for graph clustering.
    For a given graph's adjacency matrix, the spectral method computes the eigenvectors of the Laplacian matrix.
    The eigenvectors that correspond to the smallest eigenvalues are used to encode the graph.
"""

import torch

class SpectralEncoder:
    def __init__(self, n_clusters=3, *, eigen_solver='eigh', n_components=None,
                    norm_laplacian=True, random_state=None):
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
        self.norm_laplacian = norm_laplacian
        self.random_state = random_state

        # Attributes
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def __laplacian(self, A):
        r"""
            Compute the Laplacian matrix of the graph.
            L = D - A, where D is the degree matrix and A is the adjacency matrix.

            Parameters:
            -----------
                A : array-like of shape (n_nodes, n_nodes)
                    The adjacency matrix of the graph.

            Returns:
            --------
                L : array-like of shape (n_nodes, n_nodes)
                    The Laplacian matrix of the graph.
        """
        # Compute the degree matrix
        d = torch.sum(A, dim=-1)
        D = torch.diag(d)

        # Compute the Laplacian matrix
        L = D - A

        # Normalize the Laplacian matrix
        if self.norm_laplacian:
            D_sqrt_inv = torch.diag(1.0 / torch.sqrt(d))
            L = D_sqrt_inv @ L @ D_sqrt_inv

        return L

    def fit(self, A):
        r"""
            Compute the spectral eigenvectors of the Laplacian matrix of the graph.

            Parameters:
            -----------
                A : array-like of shape (n_nodes, n_nodes)
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

    def fit_transform(self, A):
        r"""
            Compute the spectral embedding of the graph.

            Parameters:
            -----------
                A : array-like of shape (n_nodes, n_nodes)
                    The adjacency matrix of the graph.

            Returns:
            --------
                embedding : array-like of shape (n_nodes, n_components)
                    The spectral embedding of the graph.
        """
        self.fit(A)
        return self.eigenvectors_

    def get_embeddings(self):
        r"""
            Get the spectral embedding of the graph.

            Returns:
            --------
                embedding : array-like of shape (n_nodes, n_components)
                    The spectral embedding of the graph.
        """
        return self.eigenvectors_


if __name__ == "__main__":
    # Example
    # Create a random graph
    A = torch.randint(0, 2, (10, 10)).float()
    A = (A + A.t()) / 2  # Make the adjacency matrix symmetric

    # Compute the spectral embedding
    spectral = SpectralEncoder(n_clusters=3, norm_laplacian=False)
    embedding = spectral.fit_transform(A)
    print(embedding)

    # With normalization
    spectral = SpectralEncoder(n_clusters=3, norm_laplacian=True)
    embedding = spectral.fit_transform(A)
    print(embedding)