from omegaconf import OmegaConf, DictConfig
from optuna import Study, Trial
import optuna
from optuna.trial import TrialState
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def process_study_cfg_parameters(cfg: DictConfig) -> DictConfig:
    """
    creates optimisation directions for study
    :param cfg: config object
    :return cfg: process config
    """
    opt_directions = []
    if len(cfg.hpo.validation_metrics) > 1:
        cfg.hpo.multi_objective_study = True
    else:
        cfg.hpo.multi_objective_study = False

    for metric in cfg.hpo.validation_metrics:
        if metric == 'nmi' or metric == 'self_f1' or metric == 'other_f1' or metric == 'modularity' :
            opt_directions.append('maximize')
        else:
            opt_directions.append('minimize')

    cfg.hpo.optimisation_directions = opt_directions
    return cfg


def sample_hyperparameters(trial: optuna.trial.Trial, args: DictConfig, prune_params=None) -> DictConfig:
    """
    iterates through the args configuration, if an item is a list then suggests a value based on
    the optuna trial instance
    :param trial: instance of trial for suggestion parameters
    :param args: config dictionary where list
    :return: new config with values replaced where list is given
    """
    vars_to_set = []
    vals_to_set = []
    for k, v in args.items_ex(resolve=False):
        if not OmegaConf.is_list(v):
            continue
        else:
            trial_sample = trial.suggest_categorical(k, OmegaConf.to_object(v))
            setattr(args, k, trial_sample)
            vars_to_set.append(k)
            vals_to_set.append(trial_sample)

    if prune_params:
        repeated = prune_params.check_params()
    for var, val in zip(vars_to_set, vals_to_set):
        print(f"model.{var}={val}")

    return args


def log_trial_result(trial: Trial, results: dict, valid_metrics: list, multi_objective_study: bool):
    """
    logs the results for a trial
    :param trial: trial object
    :param results: result dictionary for trial
    :param valid_metrics: validation metrics used
    :param multi_objective_study: boolean whether under multi-objective study or not
    """
    if not multi_objective_study:
        # log validation results
        trial_value = results[valid_metrics[0]]
        print(f'Trial {trial.number} finished. Validation result || {valid_metrics[0]}: {trial_value} ||')
        # log best trial comparison
        new_best_message = f'New best trial {trial.number}'
        if trial.number > 0:
            if trial_value > trial.study.best_value and trial.study.direction.name == 'MAXIMIZE':
                print(new_best_message)
            elif trial_value < trial.study.best_value and trial.study.direction.name == 'MINIMIZE':
                print(new_best_message)
            else:
                print(
                    f'Trial {trial.number} finished. Best trial is {trial.study.best_trial.number} with {valid_metrics[0]}: {trial.study.best_value}')
        else:
            print(new_best_message)

    else:
        # log trial results
        right_order_results = [results[k] for k in valid_metrics]
        to_log_trial_values = ''.join(
            f'| {metric}: {right_order_results[i]} |' for i, metric in enumerate(valid_metrics))
        print(f'Trial {trial.number} finished. Validation result |{to_log_trial_values}|')
        # log best trial comparison
        if trial.number > 0:
            # get best values for each metric across the best trials
            best_values, associated_trial = extract_best_trials_info(trial.study, valid_metrics)
            # compare best values for each trial with new values found
            improved_metrics = []
            for i, metric_result in enumerate(right_order_results):
                if (metric_result > best_values[i] and trial.study.directions[i].name == 'MAXIMIZE') or \
                        (metric_result < best_values[i] and trial.study.directions[i].name == 'MINIMIZE'):
                    best_values[i] = metric_result
                    improved_metrics.append(valid_metrics[i])
                    associated_trial[i] = trial.number

            # log best trial and value for each metric
            if improved_metrics:
                improved_metrics_str = ''.join(f'{metric}, ' for metric in improved_metrics)
                improved_metrics_str = improved_metrics_str[:improved_metrics_str.rfind(',')]
                print(f'New Best trial for metrics: {improved_metrics_str}')
            else:
                print('Trial worse than existing across all metrics')

            best_so_far = ''.join(
                f'trial {associated_trial[i]} ({metric}: {best_values[i]}), ' for i, metric in enumerate(valid_metrics))
            best_so_far = best_so_far[:best_so_far.rfind(',')]
            print(f'Best results so far: {best_so_far}')

    return


def extract_best_trials_info(study: Study, valid_metrics: list) -> Tuple[list, list]:
    """
    extracts the best trial from a study for each given metric and associated trial
    :param study: the study object
    :param valid_metrics: the validation metrics 
    :return best_values: the best values for each metric
    :return associated_trail: the associated trial with each best value
    """
    best_values = study.best_trials[0].values
    associated_trial = [study.best_trials[0].number] * len(valid_metrics)
    if len(study.best_trials) > 1:
        for a_best_trial in study.best_trials[1:]:
            for i, bval in enumerate(a_best_trial.values):
                if (bval > best_values[i] and study.directions[i].name == 'MAXIMIZE') or (
                        bval < best_values[i] and study.directions[i].name == 'MINIMIZE'):
                    best_values[i] = bval
                    associated_trial[i] = a_best_trial.number

    return best_values, associated_trial


class StopWhenMaxTrialsHit:
    def __init__(self, max_n_trials: int, max_n_pruned: int):
        self.max_n_trials = max_n_trials
        self.max_pruned = max_n_pruned
        self.completed_trials = 0
        self.pruned_in_a_row = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.completed_trials += 1
            self.pruned_in_a_row = 0
        elif trial.state == optuna.trial.TrialState.PRUNED:
            self.pruned_in_a_row += 1

        if self.completed_trials >= self.max_n_trials:
            print('Stopping Study for Reaching Max Number of Trials')
            study.stop()

        if self.pruned_in_a_row >= self.max_pruned:
            print('Stopping Study for Reaching Max Number of Pruned Trials in a Row')
            study.stop()


class ParamRepeatPruner:
    """Prunes repeated trials, which means trials with the same parameters won't waste time/resources."""

    def __init__(
        self,
        study: optuna.study.Study,
        repeats_max: int = 0,
        should_compare_states: List[TrialState] = [TrialState.COMPLETE],
        compare_unfinished: bool = True,
    ):
        """
        Args:
            study (optuna.study.Study): Study of the trials.
            repeats_max (int, optional): Instead of prunning all of them (not repeating trials at all, repeats_max=0) you can choose to repeat them up to a certain number of times, useful if your optimization function is not deterministic and gives slightly different results for the same params. Defaults to 0.
            should_compare_states (List[TrialState], optional): By default it only skips the trial if the paremeters are equal to existing COMPLETE trials, so it repeats possible existing FAILed and PRUNED trials. If you also want to skip these trials then use [TrialState.COMPLETE,TrialState.FAIL,TrialState.PRUNED] for example. Defaults to [TrialState.COMPLETE].
            compare_unfinished (bool, optional): Unfinished trials (e.g. `RUNNING`) are treated like COMPLETE ones, if you don't want this behavior change this to False. Defaults to True.
        """
        self.should_compare_states = should_compare_states
        self.repeats_max = repeats_max
        self.repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.unfinished_repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.compare_unfinished = compare_unfinished
        self.study = study

    @property
    def study(self) -> Optional[optuna.study.Study]:
        return self._study

    @study.setter
    def study(self, study):
        self._study = study
        if self.study is not None:
            self.register_existing_trials()

    def register_existing_trials(self):
        """In case of studies with existing trials, it counts existing repeats"""
        trials = self.study.trials
        trial_n = len(trials)
        for trial_idx, trial_past in enumerate(self.study.trials[1:]):
            self.check_params(trial_past, False, -trial_n + trial_idx)

    def prune(self):
        self.check_params()

    def should_compare(self, state):
        return any(state == state_comp for state_comp in self.should_compare_states)

    def clean_unfinised_trials(self):
        trials = self.study.trials
        finished = []
        for key, value in self.unfinished_repeats.items():
            if self.should_compare(trials[key].state):
                for t in value:
                    self.repeats[key].append(t)
                finished.append(key)

        for f in finished:
            del self.unfinished_repeats[f]

    def check_params(
        self,
        trial: Optional[optuna.trial.BaseTrial] = None,
        prune_existing=True,
        ignore_last_trial: Optional[int] = None,
    ):
        if self.study is None:
            return
        trials = self.study.trials
        if trial is None:
            trial = trials[-1]
            ignore_last_trial = -1

        self.clean_unfinised_trials()

        self.repeated_idx = -1
        self.repeated_number = -1
        for idx_p, trial_past in enumerate(trials[:ignore_last_trial]):
            should_compare = self.should_compare(trial_past.state)
            should_compare |= (
                self.compare_unfinished and not trial_past.state.is_finished()
            )
            if should_compare and trial.params == trial_past.params:
                if not trial_past.state.is_finished():
                    self.unfinished_repeats[trial_past.number].append(trial.number)
                    continue
                self.repeated_idx = idx_p
                self.repeated_number = trial_past.number
                break

        if self.repeated_number > -1:
            self.repeats[self.repeated_number].append(trial.number)
        if len(self.repeats[self.repeated_number]) > self.repeats_max:
            if prune_existing:
                print('Pruning Trial for Suggesting Duplicate Parameters')
                raise optuna.exceptions.TrialPruned()

        return self.repeated_number

    def get_value_of_repeats(
        self, repeated_number: int, func=lambda value_list: np.mean(value_list)
    ):
        if self.study is None:
            raise ValueError("No study registered.")
        trials = self.study.trials
        values = (
            trials[repeated_number].value,
            *(
                trials[tn].value
                for tn in self.repeats[repeated_number]
                if trials[tn].value is not None
            ),
        )
        return func(values)
