import random
import numpy as np
from optimizer.util.QueryProblem import get_objective_score_with_similarity

class ConexOptimizer:
    def __init__(self, file, seed=0):
        self.file = file
        self.seed = seed
        self.independent_set = file.independent_set
        self.dict_search = file.dict_search
        self.D = len(self.independent_set)
        self.history_configs = set()
        self.score_cache = {}

        self.crossover_rate = 0.5
        self.mutation_rate = 0.06
        self.pop_size = 4 * self.D

    def generate_random_config(self):
        return [random.choice(values) for values in self.independent_set]

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        crossover_indices = random.sample(range(self.D), int(self.D * self.crossover_rate))
        for idx in crossover_indices:
            child[idx] = parent2[idx]
        return child

    def mutate(self, config):
        mutated = config.copy()
        mutate_indices = random.sample(range(self.D), int(self.D * self.mutation_rate))
        for idx in mutate_indices:
            other_values = [v for v in self.independent_set[idx] if v != mutated[idx]]
            if other_values:
                mutated[idx] = random.choice(other_values)
        return mutated

    def acceptance_probability(self, current_perf, new_perf):
        if new_perf < current_perf or new_perf == 0:
            return 1.0
        else:
            return min(1.0, current_perf / new_perf)

    def run_optimizers(self, budget=20, maxlives=100):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        xs = []
        results = []
        best_result = float('inf')
        best_config = None
        best_loop = 0
        step = 0
        consecutive_no_improve = 0
        
        population = []
        while len(population) < self.pop_size:
            config = self.generate_random_config()
            current_score, similar_config = get_objective_score_with_similarity(self.dict_search, config)
            similar_config_tuple = tuple(similar_config)
            if similar_config_tuple not in self.history_configs:
                step += 1
                xs.append(similar_config)
                results.append(current_score)
                population.append((config, current_score))
                self.history_configs.add(similar_config_tuple)
                if current_score < best_result:
                    best_result = current_score
                    best_config = similar_config
                    best_loop = step
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1
            else:
                consecutive_no_improve += 1
            if step >= budget:
                break

        while step < budget and consecutive_no_improve < maxlives:
            confs_accepted = []
            for idx, (parent_config, parent_score) in enumerate(population):
                if step >= budget or consecutive_no_improve >= maxlives:
                    break
                
                current_config = self.crossover(parent_config, best_config)
                current_config = self.mutate(current_config)
                current_score, similar_config = get_objective_score_with_similarity(self.dict_search, current_config)
                similar_config_tuple = tuple(similar_config)
                
                if similar_config_tuple in self.history_configs:
                    consecutive_no_improve += 1
                    continue
                
                self.history_configs.add(similar_config_tuple)
                self.score_cache[similar_config_tuple] = current_score
                step += 1
                
                if current_score < best_result:
                    best_result = current_score
                    best_config = similar_config
                    best_loop = step
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1
                
                xs.append(similar_config)
                results.append(current_score)
                
                if step >= budget or consecutive_no_improve >= maxlives:
                    break
                
                accept_prob = self.acceptance_probability(best_result, current_score)
                if random.random() < accept_prob:
                    confs_accepted.append((current_config, current_score))
            
            if step >= budget or consecutive_no_improve >= maxlives:
                break
            
            population = confs_accepted
            
            while len(population) < self.pop_size:
                if step >= budget or consecutive_no_improve >= maxlives:
                    break
                
                config = self.generate_random_config()
                current_score, similar_config = get_objective_score_with_similarity(self.dict_search, config)
                similar_config_tuple = tuple(similar_config)
                
                if similar_config_tuple not in self.history_configs:
                    step += 1
                    xs.append(similar_config)
                    results.append(current_score)
                    population.append((config, current_score))
                    self.history_configs.add(similar_config_tuple)
                    
                    if current_score < best_result:
                        best_result = current_score
                        best_config = similar_config
                        best_loop = step
                        consecutive_no_improve = 0
                    else:
                        consecutive_no_improve += 1
                        if consecutive_no_improve >= maxlives:
                            break
                else:
                    consecutive_no_improve += 1

        used_budget = min(step, budget)
        
        return (
            xs[:used_budget],
            results[:used_budget],
            range(1, used_budget + 1),
            best_result,
            best_loop,
            used_budget
        )


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    optimizer = ConexOptimizer(file, seed=seed)
    result = optimizer.run_optimizers(budget=budget, maxlives=maxlives)
    return result