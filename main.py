from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
from typing import Tuple
import pickle
import random
import argparse
from copy import deepcopy
from ugfl.hyperoptimisation import *
from ugfl.federation import ServerModule
from optuna import Trial
from optuna.samplers import TPESampler, GridSampler
from optuna.visualization import plot_pareto_front
import json
from ugfl.datasets import create_federated_dataloaders, create_random_graph_dataloader
import os
from rich import print as rprint
import shutil


def set_random(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def hpo_objective(trial: Trial, cfg: DictConfig, model_args: DictConfig, dataloaders: Tuple, prune_params=None):
    print(f'Launching Trial {trial.number}')
    cfg.model = sample_hyperparameters(trial, deepcopy(model_args), prune_params)
    fed_server = ServerModule(cfg, dataloaders)
    if trial.number == 0:
        results = fed_server.run(return_test=False, reinit_models=True)
    else:
        results = fed_server.run(return_test=False)
    log_trial_result(trial, results, cfg.hpo.validation_metrics, cfg.hpo.multi_objective_study)

    trial.set_user_attr("f1_test_self", round(np.mean(fed_server.client.best_tests), 4))
    trial.set_user_attr("nmi_test_self", round(np.mean(fed_server.client.best_tests_nmi), 4))
    trial.set_user_attr("f1_test_other", round(np.mean(fed_server.client.best_tests_other), 4))

    if not cfg.hpo.multi_objective_study:
        return results[cfg.hpo.validation_metrics[0]]
    else:
        right_order_results = [results[k] for k in cfg.hpo.validation_metrics]
        return tuple(right_order_results)


def run(cfg: DictConfig):
    # create dataloader and set cfg from dataset
    dataloader, dataset_stats = create_federated_dataloaders(cfg.dataset.name,
                                                             cfg.dataset.ratio_train,
                                                             cfg.dataset.ratio_test,
                                                             cfg.n_clients,
                                                             cfg.dataset.partition,
                                                             cfg.dataset.batch_partition,
                                                             cfg.dataset.max_nodes_in_batch,
                                                             cfg.dataset.ratio_keep_features,
                                                             cfg.dataset.print_dataset_stats)

    OmegaConf.update(cfg, 'dataset', dataset_stats, merge=True)
    cfg.model.n_clusters = cfg.dataset.n_clusters
    cfg.model.n_nodes = max(cfg.dataset.n_nodes)
    cfg.model.n_features = cfg.dataset.n_features
    proxy_dataloader = create_random_graph_dataloader(cfg.random_graph.n_graphs,
                                                      cfg.dataset.n_features,
                                                      cfg.model.n_clusters,
                                                      cfg.random_graph.rg_type)
    dataloaders = (dataloader, proxy_dataloader)

    if cfg.hpo.analysis:
        # process study directions into multi-objective
        cfg = process_study_cfg_parameters(cfg)
        # creates store for hyperparameters
        model_args = deepcopy(cfg.model)
        optuna.logging.disable_default_handler()

        # count max trials possible
        max_n_trials_hyperopt = 1
        for k, v in model_args.items_ex(resolve=False):
            if not OmegaConf.is_list(v):
                continue
            else:
                max_n_trials_hyperopt *= len(v)
        max_n_trials_hyperopt = min(cfg.hpo.n_trials_hyperopt, max_n_trials_hyperopt)
        # creates the hpo study
        if cfg.hpo.sampler == 'tpe':
            if cfg.hpo.multi_objective_study:
                study = optuna.create_study(study_name=f'{cfg.model.name}_{cfg.dataset.name}',
                                            directions=cfg.hpo.optimisation_directions,
                                            sampler=TPESampler(seed=cfg.random_seed, multivariate=True, group=True))
            else:
                study = optuna.create_study(study_name=f'{cfg.model.name}_{cfg.dataset.name}',
                                            direction=cfg.hpo.optimisation_directions,
                                            sampler=TPESampler(seed=cfg.random_seed))

            print(f"A new hyperparameter study created: {study.study_name}")
            study_stop_cb = StopWhenMaxTrialsHit(max_n_trials_hyperopt, cfg.hpo.max_n_pruned)
            prune_params = ParamRepeatPruner(study)
            study.optimize(lambda trial: hpo_objective(trial, cfg, model_args, dataloaders, prune_params),
                           n_trials=10*cfg.hpo.n_trials_hyperopt,
                           callbacks=[study_stop_cb])

        elif cfg.hpo.sampler == 'grid':
            search_space = {}
            for k, v in model_args.items_ex(resolve=False):
                if not OmegaConf.is_list(v):
                    continue
                else:
                    search_space[k] = v

            if cfg.hpo.multi_objective_study:
                study = optuna.create_study(study_name=f'{cfg.model.name}_{cfg.dataset.name}',
                                            directions=cfg.hpo.optimisation_directions,
                                            sampler=GridSampler(search_space=search_space, seed=cfg.random_seed))
            else:
                study = optuna.create_study(study_name=f'{cfg.model.name}_{cfg.dataset.name}',
                                            direction=cfg.hpo.optimisation_directions[0],
                                            sampler=GridSampler(search_space=search_space, seed=cfg.random_seed))

            study.optimize(lambda trial: hpo_objective(trial, cfg, model_args, dataloaders, prune_params=None),
                           n_trials=max_n_trials_hyperopt)

        if not cfg.hpo.multi_objective_study:
            # print and save best hps
            rprint('[green]Best Hyperparameters: ')
            print(json.dumps(study.best_params, indent=4))
            pos = cfg.experiment_name.index('hpo')
            config_location = f'ugfl/configs/experiments/{cfg.experiment_name[:pos + 3]}_result{cfg.experiment_name[pos + 3:]}.yaml'
            OmegaConf.save(cfg, config_location)
            # print validation and test performance
            rprint(f'[green]{cfg.hpo.validation_metrics[0]} Performance: {study.best_value}')
            rprint(f'[green]{cfg.hpo.validation_metrics[0]} Test Performance: ')
            print(json.dumps(study.user_attrs, indent=4))

        else:
            for i in range(len(cfg.hpo.validation_metrics)):
                trial_with_highest = max(study.best_trials, key=lambda t: t.values[i])
                rprint(f'\n[green]Best Trial for {cfg.hpo.validation_metrics[i]} w Validation Performance: {trial_with_highest.values[i]}')
                rprint('[green]Hyperparameters: ')
                print(json.dumps(trial_with_highest.params, indent=4))
                cfg.model = OmegaConf.merge(cfg.model,
                                            OmegaConf.merge(model_args, OmegaConf.create(trial_with_highest.params)))

                pos = cfg.experiment_name.index('hpo')
                config_location = f'ugfl/configs/experiments/{cfg.experiment_name[:pos + 3]}_result{cfg.experiment_name[pos + 3:]}_{cfg.hpo.validation_metrics[i]}.yaml'
                OmegaConf.save(cfg, config_location)
                # print test performance
                rprint(f'[green]Test Performance: ')
                print(json.dumps(trial_with_highest.user_attrs, indent=4))

            fig = plot_pareto_front(study, target_names=cfg.hpo.validation_metrics)
            save_path = os.path.dirname(os.path.abspath(f'{cfg.result_path}{cfg.experiment_name}'))
            os.makedirs(save_path, exist_ok=True)
            experiment_name = os.path.basename(cfg.experiment_name)
            # save pareto front figure
            fig.write_html(f'{save_path}/{experiment_name}_pareto.html')
            # save study results
            pickle.dump(study, open(f"{save_path}/{experiment_name}_study.pkl", "wb"))

    else:
        fed_server = ServerModule(cfg, dataloaders)
        _ = fed_server.run(reinit_models=True)

        # save result tracker
        if not os.path.exists(cfg.result_path):
            os.mkdir(cfg.result_path)
        pickle.dump(fed_server.stats_tracker, open(f"{cfg.result_path}/{os.path.basename(cfg.experiment_name)}.pkl", "wb"))

    return


if __name__ == "__main__":
    # parse the command line for arguments
    parser = argparse.ArgumentParser(description='federated learning script')
    parser.add_argument('--experiment', type=str, default=None,
                        help='the name of the model to run')
    parser.add_argument('--partition', type=str, default=None,
                        help='type of split')
    parser.add_argument('--batch_partition', type=str, default=None,
                        help='type of split for dataset')
    parser.add_argument('--dataset', type=str, default=None,
                        help='the dataset')
    parser.add_argument('--n_clients', type=int, default=None,
                        help='number of clients')
    parser.add_argument('--drop_features', type=float, default=None,
                        help='percentage of features to drop')
    
    parser.add_argument('--rg_regen', action="store_true")
    parser.add_argument('--n_rgs', type=int, default=None)
    parser.add_argument('--dishonest_parties', type=int, default=None)

    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--rs', type=int, default=None)
    
    cfg = OmegaConf.load('ugfl/configs/base.yaml')

    parsed = parser.parse_args()
    override_cfg = OmegaConf.load(f'ugfl/configs/overwrite.yaml')
    cfg = OmegaConf.merge(cfg, override_cfg)
    if parsed.experiment:
        exp_cfg = OmegaConf.load(f'ugfl/configs/experiments/{parsed.experiment}.yaml')
        cfg = OmegaConf.merge(cfg, exp_cfg)
        cfg.experiment_name = parsed.experiment

    if parsed.dataset:
        cfg.dataset.name = parsed.dataset
        cfg.experiment_name += f'_{parsed.dataset}'

    if parsed.batch_partition:
        cfg.dataset.batch_partition = parsed.batch_partition

    if parsed.n_clients:
        cfg.n_clients = parsed.n_clients
        cfg.experiment_name += f'_{parsed.n_clients}'

    if parsed.partition:
        cfg.dataset.partition = parsed.partition
        cfg.experiment_name += f'_{parsed.partition}'

    if parsed.drop_features:
        cfg.dataset.ratio_keep_features = parsed.drop_features
        cfg.experiment_name += f'_dropfeatures'

    if parsed.rg_regen: 
        cfg.random_graph.regen = parsed.rg_regen
        cfg.experiment_name += f'_rg_regen'
    
    if parsed.n_rgs:
        cfg.random_graph.n_graphs = parsed.n_rgs
        cfg.experiment_name += f'_nrg{parsed.n_rgs}'

    if parsed.dishonest_parties:
        cfg.model.dishonest_parties = parsed.dishonest_parties
        cfg.experiment_name += f'_dp{parsed.dishonest_parties}'

    if parsed.gpu_id:
        cfg.gpu_id = parsed.gpu_id

    if parsed.rs:
        cfg.random_seed = int(parsed.rs)
        
    if type(cfg.random_seed) == int:
        set_random(cfg.random_seed)
        cfg.experiment_name = cfg.experiment_name + f'_{cfg.random_seed}'
        cfg.save_path = cfg.save_path + cfg.experiment_name + '/'
        run(deepcopy(cfg))
        shutil.rmtree(cfg.save_path)
    else:
        exp_name_base = deepcopy(cfg.experiment_name)
        exp_save_path = deepcopy(cfg.save_path)
        for seed in cfg.random_seed:
            set_random(seed)
            cfg.experiment_name = exp_name_base + f'_{seed}'
            cfg.save_path = exp_save_path + cfg.experiment_name + '/'
            run(deepcopy(cfg))
            shutil.rmtree(cfg.save_path)
