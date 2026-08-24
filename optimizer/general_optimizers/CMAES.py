import cma
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
    history_configs = {}
    consecutive_no_improve = 0
    step = 0
    loop = 0

    def map_to_enum_config(sol):
        new_config = []
        for i, values in enumerate(independent_set):
            index = int(abs(sol[i]) * len(values)) % len(values)
            new_config.append(values[index])
        return new_config

    initial_mean = [0.5] * len(independent_set)
    initial_sigma = 0.3
    es = cma.CMAEvolutionStrategy(
        initial_mean,
        initial_sigma,
        {'seed': seed, 'popsize': 4, 'maxiter': 1000000}
    )

    while step < budget and consecutive_no_improve < maxlives:
        solutions = es.ask()
        current_valid_sols = []
        current_valid_scores = []
        
        for sol in solutions:
            generated_config = map_to_enum_config(sol)
            current_score, similar_config = get_objective_score_with_similarity(dict_search, generated_config)
            similar_config_tuple = tuple(similar_config)
            loop += 1

            if similar_config_tuple in history_configs:
                current_score = history_configs[similar_config_tuple]
                consecutive_no_improve += 1
            else:
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

            current_valid_sols.append(sol)
            current_valid_scores.append(current_score)
            
            if step >= budget or consecutive_no_improve >= maxlives:
                break

        if len(current_valid_sols) == 4:
            es.tell(current_valid_sols, current_valid_scores)
        else:
            break
        
        if step >= budget or consecutive_no_improve >= maxlives:
            break

    return xs, results, range(1, len(xs) + 1), best_result, best_loop, step