import logging
import numpy as np
from collections import defaultdict
from hpbandster.core.nameserver import NameServer
from hpbandster.core.worker import Worker
from hpbandster.optimizers import HyperBand
import ConfigSpace as CS
from optimizer.util.QueryProblem import get_objective_score_with_similarity
import serpent

def numpy_serpent_handler(obj, serializer, outputstream, indentlevel):
    if isinstance(obj, np.integer):
        serializer._serialize(int(obj), outputstream, indentlevel)
    elif isinstance(obj, np.floating):
        serializer._serialize(float(obj), outputstream, indentlevel)
    elif isinstance(obj, np.ndarray):
        serializer._serialize(obj.tolist(), outputstream, indentlevel)
    elif isinstance(obj, np.str_):
        serializer._serialize(str(obj), outputstream, indentlevel)
    elif isinstance(obj, np.bool_):
        serializer._serialize(bool(obj), outputstream, indentlevel)
    else:
        raise serpent.Serializer.UnsupportedTypeError

serpent._special_classes_registry[np.integer] = numpy_serpent_handler
serpent._special_classes_registry[np.floating] = numpy_serpent_handler
serpent._special_classes_registry[np.ndarray] = numpy_serpent_handler
serpent._special_classes_registry[np.str_] = numpy_serpent_handler
serpent._special_classes_registry[np.bool_] = numpy_serpent_handler

def run_optimizers(file, budget=20, seed=0, maxlives=100):   
    fidelity_columns = file.fidelity_columns
    has_fidelity = bool(fidelity_columns)
    fidelities = None

    if has_fidelity:
        col = file.fidelity_columns[0]
        fidelities = sorted(file.fidelity_values[col])

    used_budget = 0
    consecutive_no_improve = 0
    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_configs = {} if not has_fidelity else defaultdict(dict)
    history_configs_2 = {} if not has_fidelity else defaultdict(dict)
    similar_configs_2 = {} if not has_fidelity else defaultdict(dict)

    class MyWorker(Worker):
        def __init__(self, file, seed, history_configs_2, similar_configs_2, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.file = file
            self.seed = seed
            self.history_configs_2 = history_configs_2
            self.similar_configs_2 = similar_configs_2
            self.task_history = []

        def compute(self, config, budget, **kwargs):
            config_list = [config[param_name] for param_name in self.file.features]
            config_list_tuple = tuple(config_list)
            fidelity = budget

            if self.file.fidelity_columns:
                if config_list_tuple in self.history_configs_2[fidelity]:
                    score = self.history_configs_2[fidelity][config_list_tuple]
                    similar_config = self.similar_configs_2[fidelity][config_list_tuple]
                else:
                    score, similar_config = get_objective_score_with_similarity(
                        self.file.dict_search[int(fidelity)],
                        config_list
                    )
                    similar_config_tuple = tuple(similar_config)
                    self.history_configs_2[fidelity][similar_config_tuple] = score
                    self.history_configs_2[fidelity][config_list_tuple] = score
                    self.similar_configs_2[fidelity][config_list_tuple] = similar_config
                    self.similar_configs_2[fidelity][similar_config_tuple] = similar_config
            else:
                if config_list_tuple in self.history_configs_2:
                    score = self.history_configs_2[config_list_tuple]
                    similar_config = self.similar_configs_2[config_list_tuple]
                else:
                    score, similar_config = get_objective_score_with_similarity(
                        self.file.dict_search,
                        config_list
                    )
                    similar_config_tuple = tuple(similar_config)
                    self.history_configs_2[similar_config_tuple] = score
                    self.history_configs_2[config_list_tuple] = score
                    self.similar_configs_2[config_list_tuple] = similar_config
                    self.similar_configs_2[similar_config_tuple] = similar_config

            self.task_history.append((similar_config, int(fidelity), score))
            return {'loss': score}

        def get_configspace(self):
            cs = CS.ConfigurationSpace(seed=self.seed)
            for idx, (param_name, param_values) in enumerate(zip(self.file.features, self.file.independent_set)):
                valid_values = [v for v in param_values]
                unique_values = sorted(list(set(valid_values)))
                cs.add(CS.CategoricalHyperparameter(
                    name=param_name,
                    choices=unique_values
                ))
            return cs

    logging.basicConfig(level=logging.WARNING)
    ns = NameServer(run_id='hpbandster_demo', host='127.0.0.1', port=None)
    ns.start()

    worker = MyWorker(
        file=file,
        seed=seed,
        history_configs_2=history_configs_2,
        similar_configs_2=similar_configs_2,
        run_id='hpbandster_demo',
        nameserver='127.0.0.1'
    )
    worker.run(background=True)

    if has_fidelity:
        col = file.fidelity_columns[0]
        fidelities = sorted(file.fidelity_values[col])
        hb = HyperBand(
            configspace=worker.get_configspace(),
            run_id='hpbandster_demo',
            nameserver='127.0.0.1',
            min_budget=fidelities[0],
            max_budget=fidelities[-1]
        )
    else:
        hb = HyperBand(
            configspace=worker.get_configspace(),
            run_id='hpbandster_demo',
            nameserver='127.0.0.1',
            min_budget=1,
            max_budget=1
        )

    processed_tasks = 0

    while used_budget < budget and consecutive_no_improve < maxlives:
        hb.run(n_iterations=1)
        new_tasks = worker.task_history[processed_tasks:]

        for i, (similar_config, fidelity, score) in enumerate(new_tasks, 1):
            similar_config_tuple = tuple(similar_config)

            if has_fidelity:
                if similar_config_tuple in history_configs[fidelity]:
                    if fidelity == fidelities[-1]:
                        consecutive_no_improve += 1
                else:
                    col = file.fidelity_columns[0]
                    fidelities = sorted(file.fidelity_values[col])
                    used_budget += fidelity / fidelities[-1]
                    history_configs[fidelity][similar_config_tuple] = score
                    config_with_fidelity_budget = similar_config + [fidelity] + [used_budget]
                    xs.append(config_with_fidelity_budget)
                    results.append(score)

                    if fidelity == fidelities[-1]:
                        if score < best_result:
                            best_result = score
                            best_loop = used_budget
                            consecutive_no_improve = 0
                        else:
                            consecutive_no_improve += 1
            else:
                if similar_config_tuple in history_configs:
                    consecutive_no_improve += 1
                else:
                    used_budget += 1
                    history_configs[similar_config_tuple] = score
                    xs.append(similar_config)
                    results.append(score)

                    if score < best_result:
                        best_result = score
                        best_loop = used_budget
                        consecutive_no_improve = 0
                    else:
                        consecutive_no_improve += 1

            if used_budget >= budget or consecutive_no_improve >= maxlives:
                break

        processed_tasks = len(worker.task_history)

    hb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return (xs, results, range(1, len(xs)+1), best_result, best_loop, used_budget)