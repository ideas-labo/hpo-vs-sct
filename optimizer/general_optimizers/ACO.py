import random
import numpy as np
from optimizer.util.QueryProblem import get_objective_score_with_similarity

def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)

    independent_set = file.independent_set
    dict_search = file.dict_search

    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_configs = {}
    consecutive_no_improve = 0
    step = 0
    loop = 0

    num_ants = 10
    alpha = 1.0
    beta = 2.0
    rho = 0.5
    Q = 100.0

    pheromone_matrix = [
        [1.0 for _ in values] 
        for values in independent_set
    ]

    while step < budget and consecutive_no_improve < maxlives:
        ant_solutions = []
        ant_scores = []
        similar_configs = []

        for _ in range(num_ants):
            solution = []
            for i, values in enumerate(independent_set):
                probabilities = []
                total = 0.0
                for j, value in enumerate(values):
                    pheromone = pheromone_matrix[i][j]
                    heuristic = 1.0
                    prob = (pheromone ** alpha) * (heuristic ** beta)
                    probabilities.append(prob)
                    total += prob
                probabilities = [p / total for p in probabilities]
                choice_idx = np.random.choice(len(values), p=probabilities)
                solution.append(values[choice_idx])

            current_score, similar_config = get_objective_score_with_similarity(dict_search, solution)
            similar_config_tuple = tuple(similar_config)

            if similar_config_tuple in history_configs:
                consecutive_no_improve += 1
                loop += 1
                continue

            history_configs[similar_config_tuple] = current_score
            step += 1
            loop += 1
            xs.append(similar_config)
            results.append(current_score)

            if current_score < best_result:
                best_result = current_score
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            ant_solutions.append(solution)
            ant_scores.append(current_score)
            similar_configs.append(similar_config)

            if step >= budget or consecutive_no_improve >= maxlives:
                break

        for i in range(len(pheromone_matrix)):
            for j in range(len(pheromone_matrix[i])):
                pheromone_matrix[i][j] *= (1 - rho)
                for k in range(len(ant_solutions)):
                    if ant_solutions[k][i] == independent_set[i][j]:
                        score = ant_scores[k] if ant_scores[k] != 0 else 1e-10
                        pheromone_matrix[i][j] += Q / score

        if consecutive_no_improve >= maxlives:
            break

    return xs, results, range(1, len(results) + 1), best_result, best_loop, step