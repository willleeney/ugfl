import torch.nn as nn
import torch
import math
from ugfl import data_processing
from sklearn.metrics import f1_score, normalized_mutual_info_score
import numpy as np
from ugfl.layers import GCN

class DMoN(nn.Module):
    def __init__(self,
                 logargs,
                 args,
                 device,
                 debug=False,
                 client_id=0,
                 progress_bar=None,
                 act='selu',
                 do_unpooling=False,
                 ):
        """Initializes the layer with specified parameters."""
        super(DMoN, self).__init__()
        self.logargs = logargs
        self.args = args
        self.debug = debug
        self.client_id = client_id
        self.n_clusters = args.n_clusters
        self.orthogonality_regularization = args.orthogonality_regularization
        self.cluster_size_regularization = args.cluster_size_regularization
        self.dropout_rate = args.dropout_rate
        self.do_unpooling = do_unpooling
        self.device = device
        self.progress_bar = progress_bar
        if self.progress_bar:
            self.model_loop = self.progress_bar.add_task(f"[cyan bold]DMoN {self.client_id+1} Loss: _ Val F1: _", total=args.n_epochs_per_round)

        self.gnn = GCN(args.n_features, args.architecture_size)
        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'selu':
            self.act = nn.SELU()

        self.classify = nn.Linear(args.architecture_size, args.n_clusters)
        self.dropout = torch.nn.Dropout(args.dropout_rate)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.classify.apply(init_weights)

        self.to(self.device)
        self.curr_rnd = 0
        self.prev_w = []
        self.class_labels = []
        for nc in range(self.n_clusters):
            self.class_labels.append(f'Cluster_{nc}')


    def training_loop(self, graph_iterator, optimizer, n_epochs):
        self.progress_bar.update(self.model_loop, total=len(graph_iterator)*n_epochs)
        self.loss_tracker = np.zeros(2)

        for epoch in range(n_epochs):
            for i, batch in enumerate(graph_iterator):
                formatted_data = data_processing.prepare_data(batch.x, batch.edge_index, self.device)
                loss = self.forward(formatted_data, extra_loss=False)

                if self.progress_bar:
                    spos = self.progress_bar.tasks[self.client_id+1].description.index('Loss: ')
                    epos = self.progress_bar.tasks[self.client_id+1].description.index('Val')
                    loop_describe = f'{self.progress_bar.tasks[self.client_id+1].description[:spos-1]} Loss: {round(loss.item(), 4)} {self.progress_bar.tasks[self.client_id+1].description[epos:]}'
                    self.progress_bar.update(self.model_loop, advance=1, description=loop_describe)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return

    def forward(self, formatted_data, extra_loss=False):
        features, graph, graph_normalised, edge_attr = formatted_data
        gnn_out = self.act(self.gnn(features, graph_normalised, edge_attr))
        assignments = self.dropout(self.classify(gnn_out))
        assignments = nn.functional.softmax(assignments, dim=1)

        n_edges = graph.nonzero().size()[0]
        n_nodes = assignments.shape[0]
        degrees = torch.sum(graph != 0., dim=0, dtype=torch.float32)
        graph_pooled = torch.spmm(torch.spmm(graph, assignments).T, assignments)
        normalizer_left = torch.matmul(assignments.permute(*torch.arange(assignments.ndim - 1, -1, -1)), degrees).unsqueeze(1)
        normalizer_right = torch.matmul(degrees, assignments).unsqueeze(0)
        normalizer = torch.spmm(normalizer_left, normalizer_right) / 2 / n_edges
        spectral_loss = - torch.trace(graph_pooled - normalizer) / 2 / n_edges
        loss = spectral_loss
        self.loss_tracker[0] += loss.item()

        if extra_loss:
            pairwise = torch.spmm(assignments.T, assignments)
            identity = torch.eye(self.n_clusters).to(pairwise.device)
            orthogonality_loss = torch.norm(pairwise / torch.norm(pairwise) -
                                            identity / math.sqrt(float(self.n_clusters)))
            orthogonality_loss *= self.orthogonality_regularization
            loss += orthogonality_loss

            cluster_loss = torch.norm(torch.sum(pairwise, dim=1)) / n_nodes * math.sqrt(
                float(self.n_clusters)) - 1
            cluster_loss *= self.cluster_size_regularization
            loss += cluster_loss

        else:
            cluster_sizes = torch.sum(assignments, dim=0)
            cluster_loss = torch.norm(cluster_sizes) / n_nodes * math.sqrt(float(self.n_clusters)) - 1
            cluster_loss *= self.cluster_size_regularization
            self.loss_tracker[1] += cluster_loss.item()
            loss += cluster_loss

        return loss

    def embed_w_grad(self, formatted_data):
        features, graph, graph_normalised, edge_attr = formatted_data
        gnn_out = self.act(self.gnn(features, graph_normalised, edge_attr))
        return gnn_out

    @torch.no_grad()
    def embed(self, formatted_data):
        features, graph, graph_normalised, edge_attr = formatted_data
        gnn_out = self.act(self.gnn(features, graph_normalised, edge_attr))
        return gnn_out

    @torch.no_grad()
    def predict(self, formatted_data):
        features, graph, graph_normalised, edge_attr = formatted_data
        gnn_out = self.act(self.gnn(features, graph_normalised, edge_attr))
        assignments = self.classify(gnn_out)
        assignments = nn.functional.softmax(assignments, dim=1)

        return assignments

    @torch.no_grad()
    def test(self, graph_iterator, testing=False, return_pred=False):
        self.eval()
        preds = []
        labels = []
        for i, batch in enumerate(graph_iterator):
            formatted_data = data_processing.prepare_data(batch.x, batch.edge_index, self.device)
            assignments = self.predict(formatted_data).argmax(axis=1)
            predictions = assignments.detach().cpu().numpy()
            preds.extend(list(predictions))
            labels.extend(list(batch.y.cpu().numpy()))

        eval_preds, _ = data_processing.hungarian_algorithm(np.array(labels), np.array(preds))
        f1 = round(f1_score(labels, eval_preds, average='macro'), 4)
        nmi = round(normalized_mutual_info_score(labels, eval_preds), 4)
        if not return_pred:
            return {'f1': f1, 'nmi': nmi}
        else:
            return eval_preds
