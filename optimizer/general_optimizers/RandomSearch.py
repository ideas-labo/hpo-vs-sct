import random
from optimizer.util.QueryProblem import get_objective_score_with_similarity

def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    independent_set = file.independent_set
    dict_search = file.dict_search
    
    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_configs = dict()
    consecutive_no_improve = 0
    step = 0
    loop = 0
    
    while step < budget and consecutive_no_improve < maxlives:
        generated_config = [random.choice(param_values) for param_values in independent_set]
        generated_config_tuple = tuple(generated_config)
        
        if generated_config_tuple in history_configs:
            consecutive_no_improve += 1
            loop += 1
            continue
        else:
            current_score, similar_config = get_objective_score_with_similarity(dict_search, generated_config)
            similar_config_tuple = tuple(similar_config)
            
            if similar_config_tuple in history_configs:
                consecutive_no_improve += 1
                loop += 1
            else:
                history_configs[similar_config_tuple] = current_score
                history_configs[generated_config_tuple] = current_score
                xs.append(similar_config)
                results.append(current_score)
                step += 1
                loop += 1
                
                if current_score < best_result:
                    best_result = current_score
                    best_loop = step
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1

    return (
        xs, 
        results, 
        range(1, step + 1),
        best_result, 
        best_loop, 
        step
    )