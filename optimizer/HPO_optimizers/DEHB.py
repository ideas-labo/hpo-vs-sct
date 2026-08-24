import random
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from dehb import DEHB
from optimizer.util.QueryProblem import get_objective_score_with_similarity
from collections import defaultdict

def run_optimizers(file, budget=20, seed=0, maxlives=100):
    fidelity_columns = file.fidelity_columns
    has_fidelity = bool(fidelity_columns)
    fidelities = None

    if has_fidelity:
        col = file.fidelity_columns[0]
        fidelities = sorted(file.fidelity_values[col])

    random.seed(seed)
    independent_set = file.independent_set
    dict_search = file.dict_search

    used_budget = 0
    consecutive_no_improve = 0
    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_configs = {} if not has_fidelity else defaultdict(dict)

    cs = ConfigurationSpace(seed=seed)
    for i, values in enumerate(independent_set):
        param = CategoricalHyperparameter(
            name=f"param_{i}",
            choices=values,
            default_value=values[0]
        )
        cs.add(param)

    dimensions = len(list(cs.values()))

    if has_fidelity:
        col = file.fidelity_columns[0]
        fidelities = sorted(file.fidelity_values[col])
        dehb = DEHB(
            cs=cs,
            dimensions=dimensions,
            min_fidelity=fidelities[0],
            max_fidelity=fidelities[-1],
            eta=3,
            seed=seed,
            n_workers=1,
            output_path="./dehb_logs"
        )
    else:
        dehb = DEHB(
            cs=cs,
            dimensions=dimensions,
            min_fidelity=1,
            max_fidelity=2,
            eta=3,
            seed=seed,
            n_workers=1,
            output_path="./dehb_logs"
        )

    while used_budget < budget and consecutive_no_improve < maxlives:
        job_info = dehb.ask()
        config = job_info["config"]
        fidelity = job_info["fidelity"]
        config_list = [config[f"param_{i}"] for i in range(len(independent_set))]
        config_list_tuple = tuple(config_list)

        if has_fidelity:
            col = file.fidelity_columns[0]
            fidelities = sorted(file.fidelity_values[col])
            if config_list_tuple in history_configs[fidelity]:
                score = history_configs[fidelity][config_list_tuple]
                if fidelity == fidelities[-1]:
                    consecutive_no_improve += 1
            else:
                score, similar_config = get_objective_score_with_similarity(dict_search[int(fidelity)], config_list)
                similar_config_tuple = tuple(similar_config)
                if similar_config_tuple in history_configs[fidelity]:
                    history_configs[fidelity][config_list_tuple] = score
                    if fidelity == fidelities[-1]:
                        consecutive_no_improve += 1
                else:
                    col = file.fidelity_columns[0]
                    fidelities = sorted(file.fidelity_values[col])
                    used_budget += fidelity / fidelities[-1]
                    history_configs[fidelity][similar_config_tuple] = score
                    history_configs[fidelity][config_list_tuple] = score
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

            dehb.tell(
                job_info=job_info,
                result={
                    "fitness": score,
                    "cost": fidelity / fidelities[-1]
                }
            )
        else:
            if config_list_tuple in history_configs:
                score = history_configs[config_list_tuple]
                consecutive_no_improve += 1
            else:
                score, similar_config = get_objective_score_with_similarity(dict_search, config_list)
                similar_config_tuple = tuple(similar_config)
                if similar_config_tuple in history_configs:
                    history_configs[config_list_tuple] = score
                    consecutive_no_improve += 1
                else:
                    used_budget += 1
                    history_configs[similar_config_tuple] = score
                    history_configs[config_list_tuple] = score
                    xs.append(similar_config)
                    results.append(score)

                    if score < best_result:
                        best_result = score
                        best_loop = used_budget
                        consecutive_no_improve = 0
                    else:
                        consecutive_no_improve += 1

            dehb.tell(
                job_info=job_info,
                result={
                    "fitness": score,
                    "cost": 1
                }
            )

        if used_budget >= budget or consecutive_no_improve >= maxlives:
            break

    return xs, results, range(1, len(xs)+1), best_result, best_loop, used_budget