import random
import numpy as np
from optimizer.util.ReadProblem import get_data
from optimizer.util.QueryProblem import get_objective_score_with_similarity

def generate_random_config(independent_set):
    return [random.choice(values) for values in independent_set]

def crossover(parent_c, parent_n):
    crossover_point = random.randint(1, len(parent_c) - 1)
    return parent_c[:crossover_point] + parent_n[crossover_point:]

def mutation(config, independent_set, mutation_rate=0.1):
    for i in range(len(config)):
        if random.random() < mutation_rate:
            values = independent_set[i]
            other_values = [v for v in values if v != config[i]]
            if other_values:
                config[i] = random.choice(other_values)
    return config

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
    history_configs = dict()
    step = 0
    loop = 0
    consecutive_no_improve = 0
    population_size = 100

    population = [
        {'config': generate_random_config(independent_set),
         'gender': random.choice(['C', 'N']),
         'age': random.randint(0, 3),
         'score': None}
        for _ in range(population_size)
    ]

    while step < budget and consecutive_no_improve < maxlives:
        for ind in population:
            if ind['gender'] == 'C' and ind['score'] is None:
                generated_config = ind['config']
                loop += 1

                current_score, similar_config = get_objective_score_with_similarity(dict_search, generated_config)
                similar_tuple = tuple(similar_config)

                if similar_tuple in history_configs:
                    consecutive_no_improve += 1
                    continue

                history_configs[similar_tuple] = current_score
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

                if step >= budget or consecutive_no_improve >= maxlives:
                    break

        if step >= budget or consecutive_no_improve >= maxlives:
            break

        competitive = [ind for ind in population if ind['gender'] == 'C' and ind['score'] is not None]
        if not competitive:
            population = [{'config': generate_random_config(independent_set),
                           'gender': random.choice(['C', 'N']),
                           'age': 0,
                           'score': None} for _ in range(population_size)]
            continue

        competitive.sort(key=lambda x: x['score'])
        num_mating = max(1, int(len(competitive) * 0.1))
        mating_c = competitive[:num_mating]

        non_competitive = [ind for ind in population if ind['gender'] == 'N']
        if not non_competitive:
            non_competitive = [{'config': generate_random_config(independent_set),
                                'gender': 'N',
                                'age': 0,
                                'score': None} for _ in range(population_size // 2)]

        offspring = []
        num_n_per_c = max(1, int(len(non_competitive) * (200/3)/100))
        for c in mating_c:
            for n in random.sample(non_competitive, num_n_per_c):
                child_config = crossover(c['config'], n['config'])
                child_config = mutation(child_config, independent_set)
                offspring.append({
                    'config': child_config,
                    'gender': random.choice(['C', 'N']),
                    'age': 0,
                    'score': None
                })
        
        for ind in population:
            ind['age'] += 1
        population = [ind for ind in population if ind['age'] <= 3]
        population.extend(offspring)
        if len(population) > population_size:
            population = random.sample(population, population_size)

        for ind in population:
            ind['score'] = None

    return (xs,
            results,
            range(1, step + 1),
            best_result,
            best_loop,
            step)