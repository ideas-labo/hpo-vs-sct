import random
import numpy as np
from optimizer.util.QueryProblem import get_objective_score_with_similarity


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)

    independent_set = file.independent_set
    dict_search = file.dict_search
    
    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []
    history_configs = {}
    consecutive_no_improve = 0
    loop = 0
    step = 0

    population_size = 10
    population = []
    similar_population = []
    
    for _ in range(population_size):
        generated_individual = [random.choice(values) for values in independent_set]
        
        try:
            score, similar_individual = get_objective_score_with_similarity(dict_search, generated_individual)
        except Exception as e:
            score = float('inf')
            similar_individual = generated_individual
        
        similar_config_tuple = tuple(similar_individual)
        
        if similar_config_tuple not in history_configs:
            history_configs[similar_config_tuple] = score
            xs.append(similar_individual)
            results.append(score)
            step += 1
            
            if step >= budget:
                break
        
        population.append(generated_individual)
        similar_population.append(similar_individual)
    
    if step > 0:
        best_result = min(results)
        best_idx = results.index(best_result)
        best_config = xs[best_idx]
        best_loop = best_idx + 1

    while step < budget and consecutive_no_improve < maxlives:
        loop += 1

        population_scores = []
        for ind in similar_population:
            ind_tuple = tuple(ind)
            population_scores.append(history_configs[ind_tuple])
        
        parents_indices = np.argsort(population_scores)[:int(population_size/2)]
        parents = [population[i] for i in parents_indices]

        parent = random.choice(parents)
        child = []
        for i, value in enumerate(parent):
            if random.random() < 0.2:
                child.append(random.choice(independent_set[i]))
            else:
                child.append(value)
        generated_config = child

        try:
            current_score, similar_config = get_objective_score_with_similarity(dict_search, generated_config)
        except Exception as e:
            current_score = float('inf')
            similar_config = generated_config
        
        similar_config_tuple = tuple(similar_config)

        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        history_configs[similar_config_tuple] = current_score
        step += 1
        xs.append(similar_config)
        results.append(current_score)

        if current_score < best_result:
            best_result = current_score
            best_config = similar_config
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

        population.append(generated_config)
        similar_population.append(similar_config)
        if len(population) > population_size:
            pop_scores = [history_configs[tuple(ind)] for ind in similar_population]
            worst_idx = np.argmax(pop_scores)
            population.pop(worst_idx)
            similar_population.pop(worst_idx)

    return xs, results, range(1, len(results) + 1), best_result, best_loop, step