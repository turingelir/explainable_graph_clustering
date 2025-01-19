from sklearn.cluster import KMeans
from ExKMC.Tree import Tree
from sklearn.datasets import make_blobs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import torch

from graspologic.simulations import sbm
from graspologic.plot import heatmap

from spectral import SpectralEncoder

def calc_cost(tree, k, x_data):
    clusters = tree.predict(x_data)
    cost = 0
    for c in range(k):
        cluster_data = x_data[clusters == c]
        if cluster_data.shape[0] > 0:
            center = cluster_data.mean(axis=0)
            for x in cluster_data:
                cost += np.linalg.norm(x - center) ** 2
    return cost

def plot_kmeans(kmeans, x_data):
    cmap = plt.cm.get_cmap('PuBuGn')

    k = kmeans.n_clusters
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                         np.arange(y_min, y_max, .1))

    values = np.c_[xx.ravel(), yy.ravel()]

    ########### K-MEANS Clustering ###########
    plt.figure(figsize=(4, 4))
    Z = kmeans.predict(values)
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=cmap,
               aspect='auto', origin='lower', alpha=0.4)

    y_kmeans = kmeans.predict(x_data)
    plt.scatter([x[0] for x in x_data], [x[1] for x in x_data], c=y_kmeans, s=20, edgecolors='black', cmap=cmap)
    for c in range(k):
        center = x_data[y_kmeans == c].mean(axis=0)
        plt.scatter([center[0]], [center[1]], c="white", marker='$%s$' % c, s=350, linewidths=.5, zorder=10,
                    edgecolors='black')

    plt.xticks([])
    plt.yticks([])
    plt.title("Near Optimal Baseline", fontsize=14)
    plt.show()
    
def plot_tree_boundary(cluster_tree, k, x_data, kmeans, plot_mistakes=False):
    cmap = plt.cm.get_cmap('PuBuGn')
    
    ########### IMM leaves ###########
    plt.figure(figsize=(4, 4))
    
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                         np.arange(y_min, y_max, .1))

    values = np.c_[xx.ravel(), yy.ravel()]
    
    y_cluster_tree = cluster_tree.predict(x_data)

    Z = cluster_tree.predict(values)
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=cmap, 
               aspect='auto', origin='lower', alpha=0.4)


    plt.scatter([x[0] for x in x_data], [x[1] for x in x_data], c=y_cluster_tree, edgecolors='black', s=20, cmap=cmap)
    for c in range(k):
        center = x_data[y_cluster_tree==c].mean(axis=0)
        plt.scatter([center[0]], [center[1]], c="white", marker='$%s$' % c, s=350, linewidths=.5, zorder=10, edgecolors='black')
        
    if plot_mistakes:
        y = kmeans.predict(x_data)
        mistakes = x_data[y_cluster_tree != y]
        plt.scatter([x[0] for x in mistakes], [x[1] for x in mistakes], marker='x', c='red', s=60, edgecolors='black', cmap=cmap)

    plt.xticks([])
    plt.yticks([])
    plt.title("Approximation Ratio: %.2f" % (cluster_tree.score(x_data) / -kmeans.score(x_data)), fontsize=14)
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=np.array(['Cluster %d' % i for i in range(len(classes))]), 
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Cluster label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_df(df, dataset, step=1, ylim=None):
    k = int(df.iloc[0].leaves)
    cols = ["CART", "KDTree", "ExKMC", "ExKMC (base: IMM)"]

    flatui = ["#3498db", "#e74c3c", "#2ecc71", "#34495e"]
    palette = sns.color_palette(flatui)

    plt.figure(figsize=(4,3))
    ax = sns.lineplot(data=df[::step][cols], linewidth=4, palette=palette, markers=True,
                      dashes=False)
    plt.yticks(fontsize=14)

    plt.xticks(np.arange(0, 1.01, 1/3) * (df.shape[0] - 1), ['$k$\n$(=%s)$' % k, 
                                                             r'$2 \cdot k$', 
                                                             r'$3 \cdot k$', 
                                                             '$4 \cdot k$\n$(=%s)$' % (4*k)], 
               fontsize=14)
    
    if ylim is not None:
        axes = plt.gca()
        axes.set_ylim(ylim)

    plt.title(dataset, fontsize=22)
    plt.xlabel("# Leaves", fontsize=18)
    ax.xaxis.set_label_coords(0.5, -0.15)
    plt.ylabel("Cost Ratio", fontsize=18)
    plt.legend(fontsize=12)
    plt.show()

if __name__ == '__main__':
    b, n_nodes, n_clusters = 1, 100, 3
    
    # Create graphs from a stochastic block model
    n = [n_nodes, n_nodes, n_nodes]
    p = [[0.6, 0.2, 0.3], 
         [0.2, 0.65, 0.2],
         [0.3, 0.2, 0.7]]

    # Print out the graph and community memberships
    print('Graph # of nodes and p values per community:')
    print('n:', n)
    print('p:', p)

    graph, community_memberships = sbm(n=n, p=p, loops=False, return_labels=True)

    print(f"Graph type: {type(graph)} | Graph shape: {graph.shape}")
    print(f"Community memberships type: {type(community_memberships)} | Community memberships shape: {community_memberships.shape}")

    # Randomize the memberships order
    community_memberships = torch.tensor(community_memberships).unsqueeze(0)
    community_memberships_p = community_memberships[:,torch.randperm(sum(n))]

    # Convert to one-hot encoding
    partition = torch.functional.F.one_hot(community_memberships_p, num_classes=n_clusters).float()

    # Display the graph
    heatmap(graph.squeeze(), title="Graph w/ 3 Clusters")

    graphTensor = torch.tensor(graph, dtype=torch.float32).unsqueeze(0)

    specEncoder = SpectralEncoder(n_components=2)
    specEncoder.fit(graphTensor)

    embeddings = specEncoder.get_embeddings().squeeze(0)
    embeddings = embeddings.reshape(embeddings.shape[1], embeddings.shape[0])

    print(embeddings.dtype)

    embeddings = embeddings.numpy().astype(np.double)

    print(embeddings.dtype)

    kmeans = KMeans(3, random_state=42)
    kmeans.fit(embeddings)

    plot_kmeans(kmeans, embeddings)

    tree_1k = Tree(3)
    tree_1k.fit(embeddings, kmeans)

    plot_tree_boundary(tree_1k, 3, embeddings, kmeans, plot_mistakes=True)