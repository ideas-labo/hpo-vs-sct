import random
import numpy as np
from optimizer.util.QueryProblem import get_objective_score_with_similarity


def iterative_first_improvement(current_config, current_score, hyperparameters, dict_search,
                                history_configs, step, budget, xs, results, best_result, best_config,
                                consecutive_no_improve, maxlives, loop, best_loop):
    param_names = list(hyperparameters.keys())
    while True:
        improved = False
        neighbors = []
        for i, param in enumerate(param_names):
            current_value = current_config[i]
            for value in hyperparameters[param]:
                if value != current_value:
                    neighbor = current_config.copy()
                    neighbor[i] = value
                    neighbors.append(neighbor)

        random.shuffle(neighbors)

        for neighbor in neighbors:
            generated_config_tuple = tuple(neighbor)
            loop += 1

            if generated_config_tuple in history_configs:
                similar_config, score = history_configs[generated_config_tuple]
                consecutive_no_improve += 1
                if consecutive_no_improve >= maxlives:
                    return (current_config, current_score, step, history_configs, xs, results,
                            best_result, best_config, consecutive_no_improve, loop, best_loop, True)
                continue

            if step >= budget:
                break

            score, similar_config = get_objective_score_with_similarity(dict_search, neighbor)
            step += 1
            history_configs[generated_config_tuple] = (similar_config, score)
            xs.append(similar_config)
            results.append(score)

            improved_flag = False
            if score < best_result:
                best_result = score
                best_config = similar_config
                current_config = neighbor
                current_score = score
                consecutive_no_improve = 0
                best_loop = step
                improved = True
                improved_flag = True
            else:
                consecutive_no_improve += 1

            if consecutive_no_improve >= maxlives:
                return (current_config, current_score, step, history_configs, xs, results,
                        best_result, best_config, consecutive_no_improve, loop, best_loop, True)

            if improved_flag:
                break

        if not improved or step >= budget:
            break

    return (current_config, current_score, step, history_configs, xs, results,
            best_result, best_config, consecutive_no_improve, loop, best_loop, False)


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)
    r = 10
    s = 3
    p_restart = 0.01

    independent_set = file.independent_set
    dict_search = file.dict_search
    hyperparameters = {f'param_{i}': values for i, values in enumerate(independent_set)}
    param_names = list(hyperparameters.keys())
    k = len(param_names)

    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []
    history_configs = dict()
    consecutive_no_improve = 0
    step = 0
    loop = 0

    initial_configs = [
        [random.choice(hyperparameters[param]) for param in param_names]
        for _ in range(r)
    ]

    for config in initial_configs:
        generated_config_tuple = tuple(config)
        loop += 1

        if generated_config_tuple in history_configs:
            similar_config, score = history_configs[generated_config_tuple]
            consecutive_no_improve += 1
            if consecutive_no_improve >= maxlives:
                break
            continue

        if step >= budget:
            break

        score, similar_config = get_objective_score_with_similarity(dict_search, config)
        step += 1
        history_configs[generated_config_tuple] = (similar_config, score)
        xs.append(similar_config)
        results.append(score)

        if score < best_result:
            best_result = score
            best_config = similar_config
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

        if consecutive_no_improve >= maxlives:
            break

    if xs:
        initial_best_idx = np.argmin(results)
        for generated_config, (similar_config, _) in history_configs.items():
            if similar_config == xs[initial_best_idx]:
                current_config = list(generated_config)
                current_score = results[initial_best_idx]
                break
        else:
            current_config = [random.choice(hyperparameters[param]) for param in param_names]
            current_score, _ = get_objective_score_with_similarity(dict_search, current_config)
    else:
        current_config = [random.choice(hyperparameters[param]) for param in param_names]
        current_score, _ = get_objective_score_with_similarity(dict_search, current_config)

    while step < budget and consecutive_no_improve < maxlives:
        (current_config, current_score, step, history_configs, xs, results,
         best_result, best_config, consecutive_no_improve, loop, best_loop, terminate) = iterative_first_improvement(
            current_config, current_score, hyperparameters, dict_search,
            history_configs, step, budget, xs, results, best_result, best_config,
            consecutive_no_improve, maxlives, loop, best_loop
        )
        if terminate:
            break

        perturbed_config = current_config.copy()
        for _ in range(s):
            param_idx = random.randint(0, k - 1)
            param = param_names[param_idx]
            current_value = perturbed_config[param_idx]
            possible_values = [v for v in hyperparameters[param] if v != current_value]
            if possible_values:
                perturbed_config[param_idx] = random.choice(possible_values)

        generated_config_tuple = tuple(perturbed_config)
        loop += 1
        if generated_config_tuple in history_configs:
            similar_config, score = history_configs[generated_config_tuple]
            consecutive_no_improve += 1
            if consecutive_no_improve >= maxlives:
                break
            continue

        if step < budget:
            score, similar_config = get_objective_score_with_similarity(dict_search, perturbed_config)
            step += 1
            history_configs[generated_config_tuple] = (similar_config, score)
            xs.append(similar_config)
            results.append(score)

            if score < best_result:
                best_result = score
                best_config = similar_config
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

        if consecutive_no_improve >= maxlives:
            break

        if random.random() < p_restart:
            current_config = [random.choice(hyperparameters[param]) for param in param_names]
            generated_config_tuple = tuple(current_config)
            loop += 1

            if generated_config_tuple in history_configs:
                similar_config, score = history_configs[generated_config_tuple]
                consecutive_no_improve += 1
                if consecutive_no_improve >= maxlives:
                    break
                continue

            if step < budget:
                current_score, similar_config = get_objective_score_with_similarity(dict_search, current_config)
                step += 1
                history_configs[generated_config_tuple] = (similar_config, current_score)
                xs.append(similar_config)
                results.append(current_score)

                if current_score < best_result:
                    best_result = current_score
                    best_config = similar_config
                    best_loop = step
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1

        if consecutive_no_improve >= maxlives:
            break

    return xs, results, range(1, step + 1), best_result, best_loop, step