import torch
from torch_geometric.datasets import WikiCS, Reddit, Planetoid, Coauthor, Flickr, AttributedGraphDataset, GitHub
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, remove_self_loops, add_remaining_self_loops, sort_edge_index, to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader, RandomNodeLoader, NeighborLoader
from torch_geometric.transforms import ToUndirected, NormalizeFeatures, Compose
import os.path as osp
import numpy as np
from ugfl.data_processing import split
import time
import scipy.sparse as sp
from math import ceil
from typing import Tuple

home_path, _ = osp.split(osp.dirname(osp.realpath(__file__)))


def load_real_dataset(dataset_name: str):
    """
    loads a real dataset given the name of the dataset
    :param dataset_name:
    :return: pytorch data object
    """
    # fetch dataset
    dataset_path = home_path + f'/data/{dataset_name}'
    undir_transform = Compose([ToUndirected(merge=True), NormalizeFeatures()])
    if dataset_name == 'WikiCS':
        dataset = WikiCS(root=dataset_path, is_undirected=True, transform=undir_transform)
    elif dataset_name == 'Reddit':
        dataset = Reddit(root=dataset_path, transform=undir_transform)
    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=dataset_path, name=dataset_name, transform=undir_transform)
    elif dataset_name in ['Photo', 'Computers']:
        dataset = Amazon(root=dataset_path, name=dataset_name, transform=undir_transform)
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=dataset_path, name=dataset_name, transform=undir_transform)
    elif dataset_name == 'Flickr':
        data = Flickr(root=dataset_path, transform=undir_transform)
    elif dataset_name in ['Facebook', 'PPI']:
        data = AttributedGraphDataset(root=dataset_path, name=dataset_name, transform=undir_transform)
    elif dataset_name == 'GitHub':
        data = GitHub(root=dataset_path, transform=undir_transform)

    data = dataset[0]
    # trick to include global node indices
    data.n_id = torch.arange(data.num_nodes)
    data.dataset_name = dataset_name
    return data


def partition_edge_index_with_node_indices(edge_index, indices):
    """
    splits an edge index given a list of nodes so that only edges with those nodes remain
    :param edge_index: list of edges
    :param indices: list of nodes
    :return: edge_index with only those nodes
    """
    return edge_index[:, torch.isin(edge_index, indices).all(dim=0)]


def random_partition(data, n_clients):
    membership = np.random.randint(low=0, high=n_clients, size=data.num_nodes)
    data.membership = torch.from_numpy(membership)
    return data


def data_feature_mask(data, n_clients, ratio_keep_features):
    n_features = int(data.x.shape[1] * ratio_keep_features)
    return torch.from_numpy(np.random.randint(low=0, high=data.x.shape[1], size=(n_clients, n_features)))


def max_nodes_in_edge_index(edge_index):
    if edge_index.nelement() == 0:
        return -1
    else:
        return int(max(edge_index.flatten()))


def create_random_graph(n_nodes, n_features, n_clusters):
    probs = (np.identity(n_clusters) * 0.1).tolist()
    cluster_sizes = split(n_nodes, n_clusters)
    edge_index = stochastic_blockmodel_graph(cluster_sizes, probs)
    features = torch.normal(mean=0, std=1, size=(n_nodes, n_features))
    labels = []
    for i in range(n_clusters):
        labels.extend([i] * cluster_sizes[i])
    labels = torch.tensor(labels)

    return edge_index, features, labels


def create_random_graph_dataloader(n_graphs: int, n_features: int, n_clusters: int, rg_type: str):
    n_nodes = 500
    # repeat n_graphs
    dataset = []
    for idx in range(1, n_graphs + 1):
        # generate a random graph
        if rg_type == 'random':
            edge_index, features, labels = create_random_graph(n_nodes=n_nodes,
                                                            n_features=n_features,
                                                            n_clusters=n_clusters)
        elif rg_type == 'easy_mode':

            probs = (np.identity(n_clusters)).tolist()
            cluster_sizes = split(n_nodes, n_clusters)
            edge_index = stochastic_blockmodel_graph(cluster_sizes, probs)

            features = np.zeros((n_nodes, n_features))
            feature_dims_fo_cluster = split(n_features, n_clusters)
            start_feat = 0
            end_feat = feature_dims_fo_cluster[0]
            start_clus = 0
            end_clus = cluster_sizes[0]
            for i in range(len(feature_dims_fo_cluster)):
                features[start_clus:end_clus, start_feat:end_feat] = np.ones_like(features[start_clus:end_clus, start_feat:end_feat])
                if i == len(feature_dims_fo_cluster) - 1:
                    break
                start_feat += feature_dims_fo_cluster[i]
                end_feat += feature_dims_fo_cluster[i+1]
                start_clus += cluster_sizes[i]
                end_clus += cluster_sizes[i+1]
            
            features = torch.Tensor(features)

            labels = []
            for i in range(n_clusters):
                labels.extend([i] * cluster_sizes[i])
            labels = torch.Tensor(np.array(labels))

        dataset.append(Data(x=features, y=labels, edge_index=edge_index))

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def numpy_to_edge_index(adjacency: np.ndarray):
    """
    converts adjacency in numpy array form to an array of active edges
    :param adjacency: input adjacency matrix
    :return adjacency: adjacency matrix update form
    """
    adj_label = sp.coo_matrix(adjacency)
    adj_label = adj_label.todok()

    outwards = [i[0] for i in adj_label.keys()]
    inwards = [i[1] for i in adj_label.keys()]

    adjacency = np.array([outwards, inwards], dtype=np.int)
    return adjacency


def dropout_edge_undirected(edge_index: torch.Tensor, p: float = 0.5, scheme='drop') -> Tuple[torch.Tensor, torch.Tensor]:
    if p <= 0. or p >= 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 -- (got {p})')
    
    undir_edge_index = edge_index[:, torch.where(edge_index[1, :] > edge_index[0, :])[0]]

    row, col = undir_edge_index
    edge_mask = torch.rand(row.size(0)) >= p
    keep_edge_index = undir_edge_index[:, edge_mask]
    if scheme == 'drop':
        drop_edge_index = undir_edge_index[:, torch.ones_like(edge_mask, dtype=bool) ^ edge_mask]
    elif scheme == 'keep':
        drop_edge_index = undir_edge_index

    keep_edge_index = torch.cat([keep_edge_index, keep_edge_index.flip(0)], dim=1)
    drop_edge_index = torch.cat([drop_edge_index, drop_edge_index.flip(0)], dim=1)

    return keep_edge_index, drop_edge_index
    

def create_federated_dataloaders(dataset_name,
                                 ratio_train=0.4,
                                 ratio_test=0.2,
                                 n_clients=10,
                                 partition='disjoint',
                                 batch_partition='none',
                                 max_nodes_in_batch=2000,
                                 ratio_keep_features=1.0,
                                 print_dataset_stats=False):
    start_time = time.time()
    ratio_val = (1 - ratio_train - ratio_test) / (1 - ratio_train)
    # load the data
    data = load_real_dataset(dataset_name=dataset_name)
    # remove self loops
    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    # useful variables
    n_clusters = len(np.unique(data.y))
    n_features = data.x.shape[1]
    nodes_in_client = np.zeros(n_clients, dtype=int)
    dataloaders = {i: {} for i in range(n_clients)}
    total_edges = data.edge_index.shape[1]
    running_edges = 0
    # partition data into clients
    if partition == "disjoint":
        cluster_data = ClusterData(data, num_parts=n_clients)
        train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)
    elif partition == "random":
        train_loader = RandomNodeLoader(data, num_parts=n_clients, shuffle=False)
    elif partition == "random_overlapping":
        dataset = []
        # get random nodes
        total_nodes = data.x.shape[0]
        nodes_per_client = ceil(total_nodes / n_clients)
        rand_nodes = torch.randperm(total_nodes)
        node_selects = torch.arange(ceil(nodes_per_client*1.5), dtype=int)
        nodes_to_add = ceil(nodes_per_client / 2)
        for i in range(n_clients):
            node_ids = rand_nodes[node_selects]
            node_ids = torch.sort(node_ids)[0]
            new_edge = partition_edge_index_with_node_indices(data.edge_index, node_ids)
            for idx, nid in enumerate(node_ids):
                new_edge = new_edge.where(new_edge != nid, idx)

            dataset.append(Data(x=data.x[node_ids, :],
                                y=data.y[node_ids],
                                edge_index=new_edge
                                ))
            node_selects += nodes_to_add
            mask = node_selects <= data.x.shape[0] - 1
            node_selects = node_selects[mask]

        # create dataloader
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    elif partition == "none":

         # drop edges for trainp
        val_edge_index, test_edge_index = dropout_edge_undirected(data.edge_index, p=ratio_test, scheme='keep')
        train_edge_index, val_edge_index = dropout_edge_undirected(val_edge_index, p=1 - (ratio_train / (1-ratio_test)), scheme='keep')
        client_splits = {'train': train_edge_index.to(dtype=torch.int32),
                         'val': val_edge_index.to(dtype=torch.int32),
                         'test': test_edge_index.to(dtype=torch.int32)}
        if print_dataset_stats:
            print(
                f'All Clients -- nodes {data.x.shape[0]} x features {data.x.shape[1]} -- edges in train {train_edge_index.shape[1]}, '
                f'val {val_edge_index.shape[1]}, test {test_edge_index.shape[1]}')

        
        for k, v in client_splits.items():
            edge_index, _ = add_remaining_self_loops(v)
            # if the end nodes have had all the edges removed then you need to manually add the final self loops
            last_node_id = data.x.shape[0]
            last_in_adj = max_nodes_in_edge_index(edge_index)
            n_ids_left = torch.arange(last_in_adj + 1, last_node_id)
            edge_index = torch.concat((edge_index, torch.stack((n_ids_left, n_ids_left))), dim=1)
            # then finally sort index
            edge_index = sort_edge_index(edge_index)
            if print_dataset_stats:
                print(f'Edges in {k} including self loops: {edge_index.shape[1]}')
            split_data = Data(x=data.x, y=data.y, edge_index=edge_index)

            for i in range(n_clients):
                dataloaders[i][k] = DataLoader([data], batch_size=1, shuffle=False)
        
        dataset_stats = {'n_nodes': [data.y.shape[0]]*n_clients,
                        'n_features': n_features,
                        'n_clusters': n_clusters,
                        'total_edges': total_edges,
                        'included_edges': total_edges,
                        'data_distributions': np.unique(data.y, return_counts=True)[1].tolist()}
        return dataloaders, dataset_stats


    for i, batch in enumerate(train_loader):
        running_edges += batch.edge_index.shape[1]
        # drop edges for trainp
        train_edge_index, test_edge_index = dropout_edge_undirected(batch.edge_index, p=1-ratio_train)
        test_edge_index, val_edge_index = dropout_edge_undirected(test_edge_index, p=ratio_val)
        client_splits = {'train': train_edge_index.to(dtype=torch.int32),
                         'val': val_edge_index.to(dtype=torch.int32),
                         'test': test_edge_index.to(dtype=torch.int32)}
        if print_dataset_stats:
            print(
                f'Client N: {i} -- nodes {batch.x.shape[0]} x features {batch.x.shape[1]} -- edges in train {train_edge_index.shape[1]}, '
                f'val {val_edge_index.shape[1]}, test {test_edge_index.shape[1]}')

        for cluster in range(n_clusters):
            n_in_cluster = int(torch.sum(batch.y == cluster))
            percent_in_cluster = round(n_in_cluster / len(batch.y), 4)
            if print_dataset_stats:
                print(f'Cluster {cluster}: {n_in_cluster} -- {percent_in_cluster}')

        for k, v in client_splits.items():
            edge_index, _ = add_remaining_self_loops(v)
            # if the end nodes have had all the edges removed then you need to manually add the final self loops
            last_node_id = batch.x.shape[0]
            last_in_adj = max_nodes_in_edge_index(edge_index)
            n_ids_left = torch.arange(last_in_adj + 1, last_node_id)
            edge_index = torch.concat((edge_index, torch.stack((n_ids_left, n_ids_left))), dim=1)
            # then finally sort index
            edge_index = sort_edge_index(edge_index)

            if ratio_keep_features != 1.0:
                n_features = int(batch.x.shape[1] * ratio_keep_features)
                feature_mask = torch.from_numpy(np.random.randint(low=0, high=batch.x.shape[1], size=(n_features)))
                features_for_client = torch.index_select(batch.x, 1, feature_mask)
            else:
                features_for_client = batch.x

            # create data object
            split_data = Data(x=features_for_client, y=batch.y, edge_index=edge_index)

            # create and save the dataloader
            if batch_partition == 'none':
                nodes_in_client[i] = batch.y.shape[0]
                dataloaders[i][k] = DataLoader(
                    [split_data],
                )
            else:
                # how many batches given a max number of nodes in batch
                batch_splits = ceil(batch.y.shape[0] / max_nodes_in_batch)
                # how many nodes are there actually in the batch
                nodes_in_client[i] = max(ceil(batch.y.shape[0] / batch_splits), nodes_in_client[i])
                if batch_partition == 'random':
                    dataloaders[i][k] = RandomNodeLoader(split_data, num_parts=batch_splits)

                elif batch_partition == 'cluster':
                    batch_data = ClusterData(split_data, num_parts=batch_splits)
                    dataloaders[i][k] = ClusterLoader(batch_data, batch_size=1, shuffle=False)

                elif batch_partition == 'neighbour':
                    dataloaders[i][k] = NeighborLoader(
                        split_data,
                        num_neighbors=[10, 10],
                        batch_size=128,
                        directed=False
                    )

    total_time = round((time.time() - start_time), 2)
    if print_dataset_stats:
        print(f'Time to Load Dataset {dataset_name}: {total_time} seconds')
        print(f'total edges {total_edges}, running count of edges {running_edges}')
    dataset_stats = {'n_nodes': [int(i) for i in nodes_in_client],
                     'n_features': n_features,
                     'n_clusters': n_clusters,
                     'total_edges': total_edges,
                     'included_edges': running_edges,
                     'data_distributions': np.unique(data.y, return_counts=True)[1].tolist()}

    return dataloaders, dataset_stats
