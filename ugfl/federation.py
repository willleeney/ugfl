import time
import numpy as np
import torch
import ugfl
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from rich.progress import Progress, TimeElapsedColumn
from rich.panel import Panel
from rich.live import Live
import re
import os
from typing import Tuple


class ServerModule:
    def __init__(self, cfg: DictConfig, dataloaders: Tuple):
        self.cfg = cfg

        self.cfg.model.n_rounds = int(self.cfg.model.total_epochs / self.cfg.model.n_epochs_per_round)

        # set devices
        self.client_device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.server_device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
        # set dataloaders
        self.dataloader = dataloaders[0]
        self.proxy_dataloader = dataloaders[1]
        # set progress bar
        self.progress_bar = None
        self.fed_loop = None
        self.client_loop = None
        self.model_loop = None

        self.cfg.model.return_dense = False

        # set client model
        self.client = None
        self.curr_rnd = 0
        self.round_begin = 0.
        self.sd = {}
        
        # fix this bullshit
        self.stats_tracker = {
            'f1_train': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'f1_self_val': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'f1_other_val': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'f1_self_test': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'f1_other_test': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'nmi_train': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'nmi_self_val': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'nmi_other_val': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'nmi_self_test': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round)),
            'nmi_other_test': np.zeros((cfg.n_clients, cfg.model.n_rounds // cfg.validation_round))
        }

        if cfg.print_config:
            print(OmegaConf.to_yaml(cfg))
            time.sleep(0.02)

    def run(self, return_test: bool = True, reinit_models: bool = False):

        self.progress_bar = Progress(*Progress.get_default_columns(), TimeElapsedColumn(), redirect_stdout=True,
                                     expand=True)
        with Live(Panel(self.progress_bar, title=f'Training {self.cfg.experiment_name}')):

            self.fed_loop = self.progress_bar.add_task("[red bold]Federation", total=self.cfg.model.n_rounds)
            self.client = ClientModule(self.cfg, 0, self.dataloader, self.proxy_dataloader, self.client_device,
                                       self.progress_bar, reinit_models)

            for curr_round in range(self.cfg.model.n_rounds):
                self.round_begin = time.time()
                # do local epoch training
                for n_client in range(self.cfg.n_clients):
                    if curr_round == 0 and n_client != 0:
                        self.client.model = self.client.create_new_model(n_client)
                        if not reinit_models:
                            self.client.load_state(load_init=True)
                    elif curr_round > 0:
                        self.client.switch_state(n_client, self.curr_rnd)
                    self.client.train_client()
                    if curr_round % self.cfg.validation_round == 0:
                        return_stats = self.client.validate()
                        for k, v in return_stats.items():
                            self.stats_tracker[k][n_client, curr_round // self.cfg.validation_round] = return_stats[k]
                    if self.cfg.model.agg_method in self.easy_fed_algos:
                        self.sd[self.client.client_id] = self.client.transfer_to_server()
                    self.client.save_state()

                # do aggregation
                self.update()
            
                if self.cfg.debug:
                    print(f'Round {self.curr_rnd} Complete. ({round(time.time() - self.round_begin, 3)}s)')
                self.curr_rnd = curr_round + 1
                # do model testing
                for n_client in range(self.cfg.n_clients):
                    self.client.switch_state(n_client, self.curr_rnd)

                test_performance = round(np.mean(np.array(self.client.best_tests)), 3)
                if self.cfg.n_clients > 1:
                    test_performance_other = round(np.mean(np.array(self.client.best_tests_other)), 3)
                else:
                    test_performance_other = 0.

                if self.cfg.debug:
                    print(f"Ave Test Performance: {test_performance}")
                # update progress bar
                self.progress_bar.update(self.fed_loop, advance=1,
                                         description=f'[red bold]Federation Self Test F1: {test_performance}, Other Test F1: {test_performance_other}')
                if curr_round != self.cfg.model.n_rounds - 1:
                    for task in self.progress_bar.task_ids[1:]:
                        self.progress_bar.reset(task)
                else:

                    for n_client in range(self.cfg.n_clients):
                        # update progress bar with best val/test f1 scores
                        descr = self.progress_bar.tasks[n_client + 1].description
                        pos = [m.start() for m in re.finditer('Val', descr)][0]
                        otherperf = self.client.best_tests_other[n_client] if self.cfg.n_clients > 1 else 0.
                        descr = f'{descr[:pos]}Best Val F1: {self.client.best_vals[n_client]}, Self Test F1: {self.client.best_tests[n_client]}, Other Test F1: {otherperf}'
                        self.progress_bar.update(n_client + 1, description=descr)

        if return_test:
            otherperf = self.client.best_tests_other[n_client] if self.cfg.n_clients > 1 else 0.
            return {'f1_self_val': round(np.mean(self.client.best_vals[self.cfg.model.dishonest_parties+1:]), 4),
                    'nmi_self_val':  round(np.mean(self.client.best_vals_nmi[self.cfg.model.dishonest_parties+1:]), 4),
                    'other_f1': otherperf}
        else:
            otherperf = round(np.mean(self.client.best_vals_other), 4) if self.cfg.n_clients > 1 else 0.
            return {'f1_self_val': round(np.mean(self.client.best_vals[self.cfg.model.dishonest_parties+1:]), 4),
                    'nmi_self_val':  round(np.mean(self.client.best_vals_nmi[self.cfg.model.dishonest_parties+1:]), 4),
                    'other_f1': otherperf}

    def aggregate(self, c_id, local_weights, ratio=None):
        st = time.time()
        aggr_theta = OrderedDict([(k, None) for k in local_weights[0].keys()])
        keys_not_aggregated = []
        if ratio is not None:
            for name, params in aggr_theta.items():
                if ((self.cfg.model.classify_aggr and name == 'classify.weight') or ('act' in name) or \
                                             ('gnn' in name and 'weight' in name and self.cfg.model.gnn_aggr) or \
                                             (self.cfg.model.bias_aggr and 'bias' in name and 'gnn' in name and self.cfg.model.gnn_aggr) or \
                                             (self.cfg.model.bias_aggr and 'bias' in name and 'classify' in name and self.cfg.model.classify_aggr)):
                    aggr_theta[name] = np.sum([theta[name] * ratio[j] for j, theta in enumerate(local_weights)], 0)
                else:
                    aggr_theta[name] = local_weights[c_id][name]
                    keys_not_aggregated.append(name)

        else:
            ratio = 1 / len(local_weights)
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
        if self.cfg.debug:
            print(f'weight aggregation done ({round(time.time() - st, 3)}s)')
        return aggr_theta, keys_not_aggregated

    def update(self):
        st = time.time()
        local_weights = []
        local_proxy_outputs = []
        local_train_sizes = []

        # upload client
        for c_id in range(self.cfg.n_clients):
            local_weights.append(self.sd[c_id]['model'].copy())
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]

        if self.cfg.debug:
            print(f'all clients have been uploaded ({time.time() - st:.2f}s)')

        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        new_model_weights, global_keys_not_agg = self.aggregate(0, local_weights, ratio)
        global_aggr_model_weights = OrderedDict(
            [(k, v) for k, v in new_model_weights.items() if k not in global_keys_not_agg])

        global_aggr_model_weights = ugfl.data_processing.convert_np_to_tensor(global_aggr_model_weights,
                                                                                with_grad=False)
        new_model_weights = ugfl.data_processing.convert_np_to_tensor(new_model_weights, with_grad=True)
        for c_id in range(self.cfg.n_clients):
            self.client.update_state(c_id, self.curr_rnd, global_aggr_model_weights, new_model_weights)

        return



class ClientModule:
    def __init__(self, cfg, client_id, dataloader, proxy_dataloader, device, progress_bar, reinit_models):
        self.optimizer = None
        self.cfg = cfg
        self.client_id = client_id
        self.dataloader = dataloader
        self.proxy_dataloader = proxy_dataloader
        self.curr_rnd = 0
        self.device = device
        self.progress_bar = progress_bar

        self.best_vals = []
        self.best_tests = []
        self.best_vals_nmi = []
        self.best_tests_nmi = []

        self.best_vals_other = []
        self.best_tests_other = []
        self.repeated_mask_names = []
        self.true_state_dict = []
        self.reinit_models = reinit_models
        if not os.path.exists(f'{self.cfg.save_path}'):
            os.makedirs(f'{self.cfg.save_path}', exist_ok=True)

        self.load_path = f'{self.cfg.save_path}client_'
        self.model = self.create_new_model(self.client_id)
        if not reinit_models:
            self.load_state(load_init=True)

    def switch_state(self, client_id, curr_rnd):
        self.client_id = client_id
        self.curr_rnd = curr_rnd
        self.load_state()

    def update_state(self, client_id, curr_rnd, aggr_local_model_weights, modulated_state_dict, only_update_prev=False):
        self.switch_state(client_id, curr_rnd)
        self.model.prev_w = aggr_local_model_weights
        if not only_update_prev:
            self.model.load_state_dict(modulated_state_dict, strict=False)
        self.save_state()

    def save_state(self, saving_best_model=False):
        if not saving_best_model:
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict(),
                'prev_w': self.model.prev_w
            }, self.load_path + f'{self.client_id}.pt')
        else:
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict(),
                'prev_w': self.model.prev_w
            }, self.load_path + f'{self.client_id}_best.pt')

    def load_state(self, load_init=False, load_best=False):
        if load_init:
            loaded = torch.load(self.load_path + f'{self.client_id}_init.pt')
        elif load_best:
            loaded = torch.load(self.load_path + f'{self.client_id}_best.pt')
        else:
            loaded = torch.load(self.load_path + f'{self.client_id}.pt')

        # remove size mismatch
        sized_mismatched_keys = []
        for k, v in self.model.state_dict().items():
            if v.size() != loaded['model'][k].size():
                set_attr(self.model, k.split("."), None)
                sized_mismatched_keys.append(k)
        self.model.load_state_dict(loaded['model'], strict=False)
        for missed_key in sized_mismatched_keys:
            set_attr(self.model, missed_key.split("."), torch.nn.Parameter(loaded['model'][missed_key]).to(self.device))

        self.model.to(self.device)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.model.prev_w = loaded['prev_w']
        for para in self.model.prev_w:
            self.model.prev_w[f'{para}'] = self.model.prev_w[f'{para}'].to(self.device)
        self.model.curr_rnd = self.curr_rnd
        self.model.client_id = self.client_id
        self.model.model_loop = self.client_id + 1

    def create_new_model(self, n_client):
        self.client_id = n_client
       
        model = ugfl.dmon.DMoN(self.cfg.log, self.cfg.model, device=self.device, progress_bar=self.progress_bar,
                                debug=self.cfg.debug, client_id=self.client_id)


        if self.cfg.model.print_module_params:
            for name, param in model.named_parameters():
                print(name, '->', param.size())

        if self.reinit_models:
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'model': model.state_dict(),
                'prev_w': model.prev_w
            }, self.load_path + f'{self.client_id}_init.pt')

        self.best_vals.append(0)
        self.best_tests.append(0)
        self.best_vals_nmi.append(0)
        self.best_tests_nmi.append(0)
        self.best_vals_other.append(0)
        self.best_tests_other.append(0)
        model.load_path = self.load_path
        return model

    def train_client(self):
        self.model.training_loop(self.dataloader[self.client_id]['train'], self.optimizer, n_epochs=self.cfg.model.n_epochs_per_round)
        return

    def validate(self):
        return_stats = {}
        # evaluate train set performance
        results = self.test(self.dataloader[self.client_id]['train'])
        return_stats['f1_train'] = results['f1']
        return_stats['nmi_train'] = results['nmi']

        # evaluate validation performance
        val_results = self.test(self.dataloader[self.client_id]['val'])
        return_stats['f1_self_val'] = val_results['f1']
        return_stats['nmi_self_val'] = val_results['nmi']

        # update progress bar
        descr = self.progress_bar.tasks[self.client_id + 1].description
        pos = [m.start() for m in re.finditer(':', descr)][1]
        descr = f"{descr[:pos + 1]} {val_results['f1']}"
        self.progress_bar.update(self.client_id + 1, description=descr)
        # evaluate test performance
        test_results = self.test(self.dataloader[self.client_id]['test'], testing=True)
        return_stats['f1_self_test'] = test_results['f1']
        return_stats['nmi_self_test'] = test_results['nmi']

        if self.cfg.n_clients > 1:
            # test performance on other clients data
            other_test_f1s = []
            for all_cid in range(self.cfg.n_clients):
                if all_cid != self.client_id:
                    
                    other_test_f1s.append(self.test(self.dataloader[all_cid]['val'])['f1'])
            return_stats['f1_other_val'] = np.mean(other_test_f1s)

            other_test_f1s = []
            for all_cid in range(self.cfg.n_clients):
                if all_cid != self.client_id:
                    other_test_f1s.append(self.test(self.dataloader[all_cid]['test'])['f1'])
            return_stats['f1_other_test'] = np.mean(other_test_f1s)

        # check if new best validation
        if val_results['f1'] > self.best_vals[self.client_id]:
            self.best_vals[self.client_id] = val_results['f1']
            self.best_tests[self.client_id] = test_results['f1']
            if self.cfg.n_clients > 1:
                self.best_vals_other[self.client_id] = round(return_stats['f1_other_val'], 4)
                self.best_tests_other[self.client_id] = round(return_stats['f1_other_test'], 4)
            self.save_state(saving_best_model=True)
        if val_results['nmi'] > self.best_vals_nmi[self.client_id]:
            self.best_vals_nmi[self.client_id] = val_results['nmi']
            self.best_tests_nmi[self.client_id] = test_results['nmi']
        
        if self.cfg.debug:
            print(f'{self.client_id}: {return_stats}')
        return return_stats

    def test(self, dataloader, testing=False, return_pred=False):
        self.model.eval()
        out = self.model.test(dataloader, testing=testing, return_pred=return_pred)
        self.model.train()
        return out


    def transfer_to_server(self):
        numpy_state_dict = ugfl.data_processing.convert_tensor_to_np(self.model.state_dict())
        return_dict = {'train_size': self.cfg.dataset.n_nodes[self.client_id]}
        return_dict['model'] = numpy_state_dict
        return return_dict

def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        return setattr(obj, names[0], val)
    else:
        return set_attr(getattr(obj, names[0]), names[1:], val)

