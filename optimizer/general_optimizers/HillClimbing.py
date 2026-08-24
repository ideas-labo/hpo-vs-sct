import random
from optimizer.util.QueryProblem import get_objective_score_with_similarity

def run_optimizers(file, budget=20, seed=0, maxlives=100):
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
    
    best_result = current_score
    best_config = similar_config
    best_loop = step
    
    current_generated_config = generated_config
    current_similar_config = similar_config
    current_score = current_score

    while step < budget and consecutive_no_improve < maxlives:
        loop += 1

        neighbor_generated_config = current_generated_config.copy()
        param_index = random.randint(0, len(neighbor_generated_config) - 1)
        neighbor_generated_config[param_index] = random.choice(independent_set[param_index])

        neighbor_score, neighbor_similar_config = get_objective_score_with_similarity(dict_search, neighbor_generated_config)
        neighbor_similar_tuple = tuple(neighbor_similar_config)

        if neighbor_similar_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        history_configs[neighbor_similar_tuple] = neighbor_score
        step += 1
        xs.append(neighbor_similar_config)
        results.append(neighbor_score)
        
        if neighbor_score < best_result:
            best_result = neighbor_score
            best_config = neighbor_similar_config
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        if neighbor_score < current_score:
            current_generated_config = neighbor_generated_config
            current_similar_config = neighbor_similar_config
            current_score = neighbor_score

    return xs, results, range(1, len(results) + 1), best_result, best_loop, step