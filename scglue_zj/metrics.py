r"""
Performance evaluation metrics
"""

from typing import Tuple
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
import sklearn.metrics
import sklearn.neighbors
from anndata import AnnData
from scipy.sparse.csgraph import connected_components
import torch.nn as nn
from .typehint import RandomState
from .utils import get_rs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def mean_average_precision(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Mean average precision

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    neighbor_frac
        Nearest neighbor fraction
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    map
        Mean average precision
    """
    k = max(round(y.shape[0] * neighbor_frac), 1)
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), **kwargs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    return np.apply_along_axis(_average_precision, 1, match).mean().item()


def _average_precision(match: np.ndarray) -> float:
    if np.any(match):
        cummean = np.cumsum(match) / (np.arange(match.size) + 1)
        return cummean[match].mean().item()
    return 0.0


def normalized_mutual_info(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Normalized mutual information with true clustering

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.normalized_mutual_info_score`

    Returns
    -------
    nmi
        Normalized mutual information

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    x = AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    nmi_list = []
    for res in (np.arange(20) + 1) / 10:
        sc.tl.leiden(x, resolution=res)
        leiden = x.obs["leiden"]
        nmi_list.append(sklearn.metrics.normalized_mutual_info_score(
            y, leiden, **kwargs
        ).item())
    return max(nmi_list)


def avg_silhouette_width(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Cell type average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_score`

    Returns
    -------
    asw
        Cell type average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    return (sklearn.metrics.silhouette_score(x, y, **kwargs).item() + 1) / 2


def graph_connectivity(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> float:
    r"""
    Graph connectivity

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`scanpy.pp.neighbors`

    Returns
    -------
    conn
        Graph connectivity
    """
    x = AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X", **kwargs)
    conns = []
    for y_ in np.unique(y):
        x_ = x[y == y_]
        _, c = connected_components(
            x_.obsp['connectivities'],
            connection='strong'
        )
        counts = pd.value_counts(c)
        conns.append(counts.max() / counts.sum())
    return np.mean(conns).item()


def seurat_alignment_score(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01,
        n_repeats: int = 4, random_state: RandomState = None, **kwargs
) -> float:
    r"""
    Seurat alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = get_rs(random_state)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k + 1, **kwargs
        ).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.mean(repeat_scores).item()


def avg_silhouette_width_batch(
        x: np.ndarray, y: np.ndarray, ct: np.ndarray, **kwargs
) -> float:
    r"""
    Batch average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    ct
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_samples`

    Returns
    -------
    asw_batch
        Batch average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    s_per_ct = []
    for t in np.unique(ct):
        mask = ct == t
        try:
            s = sklearn.metrics.silhouette_samples(x[mask], y[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
    return np.mean(s_per_ct).item()


def neighbor_conservation(
        x: np.ndarray, y: np.ndarray, batch: np.ndarray,
        neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Neighbor conservation score

    Parameters
    ----------
    x
        Cooordinates after integration
    y
        Coordinates before integration
    b
        Batch
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    nn_cons
        Neighbor conservation score
    """
    nn_cons_per_batch = []
    for b in np.unique(batch):
        mask = batch == b
        x_, y_ = x[mask], y[mask]
        k = max(round(x.shape[0] * neighbor_frac), 1)
        nnx = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(x_.shape[0], k + 1), **kwargs
        ).fit(x_).kneighbors_graph(x_)
        nny = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(y_.shape[0], k + 1), **kwargs
        ).fit(y_).kneighbors_graph(y_)
        nnx.setdiag(0)  # Remove self
        nny.setdiag(0)  # Remove self
        n_intersection = nnx.multiply(nny).sum(axis=1).A1
        n_union = (nnx + nny).astype(bool).sum(axis=1).A1
        nn_cons_per_batch.append((n_intersection / n_union).mean())
    return np.mean(nn_cons_per_batch).item()


def foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    fracs = []

    # 下边都是给错误点数求平均
    for i in range(len(foscttm_x)):
        fracs.append((foscttm_x[i] + foscttm_y[i]) / 2)
    return np.mean(fracs).round(4)


import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import silhouette_samples, silhouette_score


def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score

    Algorithm
    ---------
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.

    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.

    Returns
    -------
    Batch entropy mixing score
    """

    #     print("Start calculating Entropy mixing score")
# def entropy(batches):
#         p = np.zeros(N_batches)
#         adapt_p = np.zeros(N_batches)
#         a = 0
#         for i in range(N_batches):
#             p[i] = np.mean(batches == batches_[i])
#             a = a + p[i] / P[i]
#         entropy = 0
#         for i in range(N_batches):
#             adapt_p[i] = (p[i] / P[i]) / a
#             entropy = entropy - adapt_p[i] * np.log(adapt_p[i] + 10 ** -8)
#         return entropy
#
#     n_neighbors = min(n_neighbors, len(data) - 1)
#     nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
#     nne.fit(data)
#     kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])
#
#     score = 0
#     batches_ = np.unique(batches)
#     N_batches = len(batches_)
#     if N_batches < 2:
#         raise ValueError("Should be more than one cluster for batch mixing")
#     P = np.zeros(N_batches)
#     for i in range(N_batches):
#         P[i] = np.mean(batches == batches_[i])
#     for t in range(n_pools):
#         indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
#         score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
#         [kmatrix[indices].nonzero()[0] == i]])
#                           for i in range(n_samples_per_pool)])
#     Score = score / float(n_pools)
#     return Score / float(np.log2(N_batches))


# def silhouette(
#         X,
#         cell_type,
#         metric='euclidean',
#         scale=True
# ):
#     """
#     Wrapper for sklearn silhouette function values range from [-1, 1] with
#         1 being an ideal fit
#         0 indicating overlapping clusters and
#         -1 indicating misclassified cells
#     By default, the score is scaled between 0 and 1. This is controlled `scale=True`
#
#     :param group_key: key in adata.obs of cell labels
#     :param embed: embedding key in adata.obsm, default: 'X_pca'
#     :param scale: default True, scale between 0 (worst) and 1 (best)
#     """
#     asw = silhouette_score(
#         X,
#         cell_type,
#         metric=metric
#     )
#     if scale:
#         asw = (asw + 1) / 2
#     return asw


# def label_transfer(ref, query, rep='latent', label='celltype'):
#     """
#     Label transfer
#
#     Parameters
#     -----------
#     ref
#         reference containing the projected representations and labels
#     query
#         query data to transfer label
#     rep
#         representations to train the classifier. Default is `latent`
#     label
#         label name. Defautl is `celltype` stored in ref.obs
#
#     Returns
#     --------
#     transfered label
#     """
#
#     from sklearn.neighbors import KNeighborsClassifier
#
#     X_train = ref.obsm[rep]
#     y_train = ref.obs[label]
#     X_test = query.obsm[rep]
#
#     knn = knn = KNeighborsClassifier().fit(X_train, y_train)
#     y_test = knn.predict(X_test)
#
#     return y_test

# def kmeans_loss(self, z):
#     z = z.to('cuda:0')
#     self.mu = nn.Parameter(self.mu.cuda())
#     #print(z.is_cuda, self.mu.is_cuda)
#     dist1 = self.tau*torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2)
#     temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
#     q = torch.exp(-temp_dist1)
#     q = (q.t() / torch.sum(q, dim=1)).t()
#     q = torch.pow(q, 2)
#     q = (q.t() / torch.sum(q, dim=1)).t()
#     dist2 = dist1 * q
#     return dist1, torch.mean(torch.sum(dist2, dim=1))

def silhouette(
        X,
        cell_type,
        metric='euclidean',
        scale=True
):
    """
    Wrapper for sklearn silhouette function values range from [-1, 1] with
        1 being an ideal fit
        0 indicating overlapping clusters and
        -1 indicating misclassified cells
    By default, the score is scaled between 0 and 1. This is controlled `scale=True`

    :param group_key: key in adata.obs of cell labels
    :param embed: embedding key in adata.obsm, default: 'X_pca'
    :param scale: default True, scale between 0 (worst) and 1 (best)
    """
    asw = silhouette_score(
        X,
        cell_type,
        metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw


# def label_transfer(ref, query, rep='latent', label='cell_type'):
#     """
#     Label transfer
#
#     Parameters
#     -----------
#     ref
#         reference containing the projected representations and labels
#     query
#         query data to transfer label
#     rep
#         representations to train the classifier. Default is `latent`
#     label
#         label name. Defautl is `celltype` stored in ref.obs
#
#     Returns
#     --------
#     transfered label
#     """
#
#     from sklearn.neighbors import KNeighborsClassifier
#
#     X_train = ref.obsm[rep]
#     y_train = ref.obs[label]
#     X_test = query.obsm[rep]
#
#     knn = knn = KNeighborsClassifier().fit(X_train, y_train)
#     y_test = knn.predict(X_test)
#
#     return y_test
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
def label_transfer(labels_true, labels_pred):
    # 从元组中提取细胞类型，假设细胞类型是每个元组的第二个元素
    # labels_true_celltype = labels_true.apply(lambda x: x[1])

    # 将真实标签转换为数值型
    # unique_labels = labels_true_celltype.unique()
    labels_true = labels_true.apply(lambda x: x.split(',')[-1].strip())

    # 将真实标签转换为数值型
    unique_labels = labels_true.unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels_true_mapped = labels_true.map(label_mapping).to_numpy(dtype=int)
    # 确保labels_true_mapped是纯数值型numpy数组
    # labels_true_num = labels_true_mapped.to_numpy(dtype=int)
    # 构建成本矩阵
    max_pred = labels_pred.max()
    max_true = labels_true_mapped.max()
    size = max(max_pred, max_true) + 1
    cost_matrix = np.zeros((size, size), dtype=np.int64)
    for i in range(size):
        for j in range(size):
            cost_matrix[i, j] = np.sum((labels_pred == i) & (labels_true_mapped == j))

    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    # 调整预测标签
    new_labels_pred = np.zeros_like(labels_pred)
    for i in range(len(row_ind)):
        new_labels_pred[labels_pred == row_ind[i]] = col_ind[i]
    acc = accuracy_score(labels_true_mapped, new_labels_pred)
    return acc

import anndata as ad
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from typing import Iterable
from tqdm import tqdm
def embedding_leiden_across_resolutions(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int,
    resolutions: Iterable[float] = np.arange(0.1, 2.1, 0.1),
):
    """Compute the ARI and NMI for Leiden clustering on an embedding,
    for various resolutions.

    Args:
        embedding (np.ndarray):
            The embedding, shape (n_obs, n_latent)
        labels (np.ndarray):
            The labels, shape (n_obs,)
        n_neighbors (int):
            The number of neighbors to use for the kNN graph.
        resolutions (Iterable[float], optional):
            The resolutions to use for Leiden clustering. Defaults to
            np.arange(0.1, 2.1, 0.1).

    Returns:
        Tuple[Iterable[float], Iterable[float], Iterable[float]]:
            The resolutions, ARIs and NMIs.
    """
    # Create an AnnData object with the joint embedding.
    joint_embedding = ad.AnnData(embedding)

    # Initialize the results.
    aris, nmis = [], []

    # Compute neighbors on the joint embedding.
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    # For all resolutions,
    for resolution in tqdm(resolutions):

        # Perform Leiden clustering.
        sc.tl.leiden(joint_embedding, resolution=resolution)

        # Compute ARI and NMI
        aris.append(ARI(joint_embedding.obs["leiden"], labels))
        nmis.append(NMI(joint_embedding.obs["leiden"], labels))

    # Return ARI and NMI for various resolutions.
    return resolutions, aris, nmis

from scipy.spatial.distance import cdist
def embedding_to_knn(
    embedding: np.ndarray, k: int = 15, metric: str = "euclidean"
) -> np.ndarray:
    """Convert embedding to knn

    Args:
        embedding (np.ndarray): The embedding (n_obs, n_latent)
        k (int, optional): The number of nearest neighbors. Defaults to 15.
        metric (str, optional): The metric to compute neighbors with. Defaults to "euclidean".

    Returns:
        np.ndarray: The knn (n_obs, k)
    """
    # Initialize the knn graph.
    knn = np.zeros((embedding.shape[0], k), dtype=int)

    # Compute pariwise distances between observations.
    distances = cdist(embedding, embedding, metric=metric)

    # Iterate over observations.
    for i in range(distances.shape[0]):

        # Get the `max_neighbors` nearest neighbors.
        knn[i] = distances[i].argsort()[1 : k + 1]

    # Return the knn graph.
    return knn

def knn_purity_score(knn: np.ndarray, labels: np.ndarray) -> float:
    """Compute the kNN purity score, averaged over all observations.
    For one observation, the purity score is the percentage of
    nearest neighbors that share its label.

    Args:
        knn (np.ndarray):
            The knn, shaped (n_obs, k). The i-th row should contain integers
            representing the indices of the k nearest neighbors.
        labels (np.ndarray):
            The labels, shaped (n_obs)

    Returns:
        float: The purity score.
    """
    # Check the dimensions of the input.
    assert knn.shape[0] == labels.shape[0]

    # Initialize a list of purity scores.
    score = 0

    # Iterate over the observations.
    for i, neighbors in enumerate(knn):

        # Do the neighbors have the same label as the observation?
        matches = labels[neighbors] == labels[i]

        # Add the purity rate to the scores.
        score += np.mean(matches) / knn.shape[0]

    # Return the average purity.
    return score
