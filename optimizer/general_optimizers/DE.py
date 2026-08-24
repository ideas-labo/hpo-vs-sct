import random
import numpy as np
from optimizer.util.ReadProblem import get_data
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
    loop = 0
    step = 0

    pop_size = 10
    CR = 0.7

    population = []
    similar_population = []
    init_scores = []
    while len(population) < pop_size:
        generated_individual = [random.choice(values) for values in independent_set]
        loop += 1
        
        score, similar_individual = get_objective_score_with_similarity(dict_search, generated_individual)
        similar_config_tuple = tuple(similar_individual)
        
        if similar_config_tuple not in history_configs:
            population.append(generated_individual)
            similar_population.append(similar_individual)
            init_scores.append(score)
            history_configs[similar_config_tuple] = score
            xs.append(similar_individual)
            results.append(score)
            step += 1
            
            if score < best_result:
                best_result = score
                best_config = similar_individual
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
        else:
            consecutive_no_improve += 1

    if step >= budget:
        return xs[:budget], results[:budget], range(1, step + 1), best_result, best_loop, step

    while step < budget and consecutive_no_improve < maxlives:
        for i in range(pop_size):
            if step >= budget or consecutive_no_improve >= maxlives:
                break

            indices = [j for j in range(pop_size) if j != i]
            a, b, c = random.sample(indices, 3)

            mutant = []
            for j in range(len(population[i])):
                possible_values = [v for v in independent_set[j] if v != population[i][j]]
                if possible_values:
                    mutant_val = random.choice(possible_values)
                else:
                    mutant_val = population[i][j]
                mutant.append(mutant_val)

            generated_trial = []
            for j in range(len(population[i])):
                if random.random() < CR or j == random.randint(0, len(population[i]) - 1):
                    generated_trial.append(mutant[j])
                else:
                    generated_trial.append(population[i][j])

            loop += 1

            trial_score, similar_trial = get_objective_score_with_similarity(dict_search, generated_trial)
            similar_trial_tuple = tuple(similar_trial)

            if similar_trial_tuple in history_configs:
                consecutive_no_improve += 1
                continue
            else:
                step += 1
                history_configs[similar_trial_tuple] = trial_score
                xs.append(similar_trial)
                results.append(trial_score)

                current_score = init_scores[i]
                if trial_score < current_score:
                    population[i] = generated_trial
                    similar_population[i] = similar_trial
                    init_scores[i] = trial_score

                if trial_score < best_result:
                    best_result = trial_score
                    best_config = similar_trial
                    best_loop = step
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1

        if consecutive_no_improve >= maxlives:
            break

    return xs[:budget], results[:budget], range(1, step + 1), best_result, best_loop, step