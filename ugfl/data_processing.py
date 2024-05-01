import torch
import scipy.sparse as sp
from torch_geometric.utils import to_torch_coo_tensor, to_scipy_sparse_matrix, dense_to_sparse, \
    from_scipy_sparse_matrix, sort_edge_index, add_remaining_self_loops
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import numpy as np
from torch_geometric.utils import to_dense_adj
from scipy.stats import wasserstein_distance


def convert_state_skip_mask(state_dict, model):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'bias' in k or 'act' in k:
            _state_dict[k] = v.clone().detach().cpu().requires_grad_(True)
        else:
            mask_key_name = k.split(".")[0] + ".mask"
            _state_dict[mask_key_name] = model[mask_key_name].clone().detach().cpu().requires_grad_(True)
            _state_dict[k] = v * _state_dict[mask_key_name]

    return _state_dict


def convert_tensor_to_np(state_dict):
    # returns the state dictionary as a numpy array
    return OrderedDict([(k, v.clone().detach().cpu().numpy()) for k, v in state_dict.items()])


def convert_np_to_tensor(state_dict, with_grad=False):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        if with_grad:
            _state_dict[k] = torch.tensor(v).requires_grad_(True)
        else:
            _state_dict[k] = torch.tensor(v)
    return _state_dict



def prepare_data(features, adjacency, device):
    features[features != 0] = 1.
    features = features.to(device)
    #adjacency, _ = add_remaining_self_loops(adjacency)
    graph = to_dense_adj(adjacency).squeeze(0)

    # normalise adj
    graph_normalised = to_scipy_sparse_matrix(adjacency)
    graph_normalised = normalize_adj(graph_normalised)
    graph_normalised, edge_attr = from_scipy_sparse_matrix(graph_normalised)

    # transfer to device
    graph = graph.to(device)
    graph_normalised = graph_normalised.to(device)
    edge_attr = edge_attr.to(device)

    data = (features, graph, graph_normalised, edge_attr)

    return data



def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split(n, k):
    d, r = divmod(n, k)
    return [d + 1] * r + [d] * (k - r)


def hungarian_algorithm(labels: np.ndarray, preds: np.ndarray, col_ind=None):
    """
    hungarian algorithm for prediction reassignment
    """
    labels = labels.astype(np.int64)
    assert preds.size == labels.size
    D = max(preds.max(), labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(preds.size):
        w[preds[i], labels[i]] += 1

    if col_ind is None:
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        preds = col_ind[preds]

    else:
        preds = [col_ind[int(i)] for i in preds]
        preds = np.asarray(preds)

    return preds, col_ind


def wasserstein_randomness(result_object):
    #  W coefficient from wasserstein where draws are the average rank between those tied
    # result_object shape [tests, seeds, algorithms]
    wills_order = []
    all_ranks = np.zeros_like(result_object)
    for t, test in enumerate(result_object):
        rank_test = np.zeros_like(test)
        for rs, rs_test in enumerate(test):
            rank_test[rs] = rank_scores(rs_test)
        wills_order.append(w_rand_wasserstein(rank_test))
        all_ranks[t] = rank_test
    wills_order = np.array(wills_order)
    algo_ranks = np.transpose(all_ranks, (2, 0, 1))
    algo_ranks = np.mean(algo_ranks.reshape(algo_ranks.shape[0], algo_ranks.shape[1] * algo_ranks.shape[2]), axis=1)
    return np.mean(wills_order), algo_ranks

def nrule(n):
    j = 0 
    for i in range(1,n+1):
        j += (i*(i-1))/2
    return j

def w_rand_wasserstein(rankings):
    n_algorithms = rankings.shape[1]
    # rank_test[:, 0] -> all seeds, one algorithm
    wass_agg = []
    for i in range(n_algorithms):
        for j in range(i):  # Iterate up to i to stay to the left of the diagonal
            wass_agg.append(wasserstein_distance(rankings[:, i], rankings[:, j]))
    return 1 - (np.sum(wass_agg) / nrule(n_algorithms))

def rank_scores(scores):
    # Get indices in descending order
    indices = np.flip(np.argsort(scores))

    # Initialize an array to store the ranks
    ranks = np.zeros_like(indices, dtype=float)

    # Assign ranks to the sorted indices
    ranks[indices] = np.arange(len(scores)) + 1

    # Find unique scores and their counts
    unique_scores, counts = np.unique(scores, return_counts=True)

    # Calculate mean ranks for tied scores
    for score, count in zip(unique_scores, counts):
        if count > 1:
            score_indices = np.where(scores == score)[0]
            mean_rank = np.mean(ranks[score_indices])
            ranks[score_indices] = mean_rank

    return ranks