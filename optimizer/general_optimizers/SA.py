import random
import math
from optimizer.util.QueryProblem import get_objective_score_with_similarity


def run_optimizers(file, budget=200, seed=0, maxlives=100):
    random.seed(seed)
    independent_set = file.independent_set
    dict_search = file.dict_search

    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []
    history_configs = dict()
    consecutive_no_improve = 0
    step = 0
    loop = 0

    generated_config = [random.choice(values) for values in independent_set]
    current_score, similar_config = get_objective_score_with_similarity(dict_search, generated_config)
    similar_config_tuple = tuple(similar_config)
    history_configs[similar_config_tuple] = current_score
    xs.append(similar_config)
    results.append(current_score)
    step += 1
    loop += 1
    best_result = current_score
    best_config = similar_config
    best_loop = step

    current_generated_config = generated_config
    current_similar_config = similar_config
    current_score = current_score

    initial_temperature = 200
    final_temperature = 0.01

    while step < budget and consecutive_no_improve < maxlives:
        temperature = initial_temperature * (1 - (step / budget)) **2 if budget != 0 else final_temperature

        neighbor_generated_config = current_generated_config.copy()
        param_index = random.randint(0, len(independent_set) - 1)
        neighbor_generated_config[param_index] = random.choice(independent_set[param_index])

        neighbor_score, neighbor_similar_config = get_objective_score_with_similarity(dict_search, neighbor_generated_config)
        neighbor_similar_tuple = tuple(neighbor_similar_config)

        if neighbor_similar_tuple in history_configs:
            consecutive_no_improve += 1
            loop += 1
            continue
        
        history_configs[neighbor_similar_tuple] = neighbor_score
        step += 1
        loop += 1
        xs.append(neighbor_similar_config)
        results.append(neighbor_score)

        if (neighbor_score < current_score) or (random.random() < math.exp((current_score - neighbor_score) / temperature)):
            current_generated_config = neighbor_generated_config
            current_similar_config = neighbor_similar_config
            current_score = neighbor_score
        
        if current_score < best_result:
            best_result = current_score
            best_config = current_similar_config
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return (
        xs, 
        results, 
        range(1, len(results) + 1),
        best_result, 
        best_loop, 
        step
    )