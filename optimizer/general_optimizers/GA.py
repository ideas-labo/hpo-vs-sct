import random
import numpy as np
from optimizer.util.ReadProblem import get_data
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
    global_step = 0

    population_size = 10
    mutation_rate = 0.3

    population = [
        [random.choice(param_values) for param_values in independent_set]
        for _ in range(population_size)
    ]

    while global_step < budget and consecutive_no_improve < maxlives:
        fitness_scores = []
        new_population = []
        similar_new_population = []

        for generated_config in population:
            loop += 1
            
            current_score, similar_config = get_objective_score_with_similarity(dict_search, generated_config)
            similar_config_tuple = tuple(similar_config)

            if similar_config_tuple in history_configs:
                consecutive_no_improve += 1    
                if consecutive_no_improve >= maxlives:
                    return xs, results, range(1, len(results)+1), best_result, best_loop, global_step
                continue

            history_configs[similar_config_tuple] = current_score
            global_step += 1

            if current_score < best_result:
                best_result = current_score
                best_config = similar_config
                best_loop = global_step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            if global_step >= budget or consecutive_no_improve >= maxlives:
                return xs, results, range(1, len(results)+1), best_result, best_loop, global_step

            xs.append(similar_config)
            results.append(current_score)
            fitness_scores.append(current_score)
            new_population.append(generated_config)
            similar_new_population.append(similar_config)

        if len(new_population) == 0:
            population = [
                [random.choice(param_values) for param_values in independent_set]
                for _ in range(population_size)
            ]
            continue

        fitness_values = [1 / max(score, 1e-10) for score in fitness_scores]
        total_fitness = sum(fitness_values)
        selection_probs = [f / total_fitness for f in fitness_values] if total_fitness > 0 else [1/len(fitness_values)]*len(fitness_values)
        selection_probs[-1] = 1.0 - sum(selection_probs[:-1])
        indices = np.arange(len(new_population))
        selected_indices = np.random.choice(indices, size=population_size, p=selection_probs)
        selected = [new_population[i] for i in selected_indices]

        next_generation = []
        for i in range(0, population_size, 2):
            if i + 1 >= population_size:
                next_generation.append(selected[i])
                break
            parent1, parent2 = selected[i], selected[i+1]
            crossover_point = random.randint(1, len(independent_set)-1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            next_generation.extend([child1, child2])

        for config in next_generation:
            for i in range(len(config)):
                if random.random() < mutation_rate:
                    config[i] = random.choice(independent_set[i])

        population = next_generation[:population_size]

    return xs, results, range(1, len(results)+1), best_result, best_loop, global_step