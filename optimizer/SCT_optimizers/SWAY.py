import random
import numpy as np
from scipy.spatial.distance import euclidean
from optimizer.util.QueryProblem import get_objective_score_with_similarity


class SWAYOptimizer:
    def __init__(self, file, seed=0, maxlives=100):
        self.file = file
        self.seed = seed
        self.maxlives = maxlives
        self.independent_set = file.independent_set
        self.dict_search = file.dict_search
        self.D = len(self.independent_set)
        self.history_configs = dict()
        self.initial_pop = []
        self.xs = []
        self.results = []

        self.enough = None
        self.total_group = 10

        self.discrete_maps = self._build_discrete_maps()

    def _build_discrete_maps(self):
        maps = []
        for vals in self.independent_set:
            maps.append({v: idx for idx, v in enumerate(vals)})
        return maps

    def _convert_to_int(self, item):
        converted = []
        for i, val in enumerate(item):
            if val not in self.discrete_maps[i]:
                raise ValueError(f"{val}")
            converted.append(self.discrete_maps[i][val])
        return converted

    def _split_items(self, items):
        if not items:
            return self._get_valid_rep(items), self._get_valid_rep(items), [], []

        rand_item = self._get_valid_rep(items)
        rand_int = self._convert_to_int(rand_item)
        east = max(items, key=lambda x: euclidean(self._convert_to_int(x), rand_int))
        east_int = self._convert_to_int(east)
        west = max(items, key=lambda x: euclidean(self._convert_to_int(x), east_int))

        c = euclidean(east_int, self._convert_to_int(west))
        if c == 0:
            mid = len(items) // 2
            return west, east, items[:mid], items[mid:]

        projections = []
        for item in items:
            item_int = self._convert_to_int(item)
            a = euclidean(item_int, self._convert_to_int(west))
            b = euclidean(item_int, east_int)
            proj = (a ** 2 + c ** 2 - b ** 2) / (2 * c) if c != 0 else 0
            projections.append((item, proj))

        projections.sort(key=lambda x: x[1])
        mid = len(projections) // 2
        return west, east, [p[0] for p in projections[:mid]], [p[0] for p in projections[mid:]]

    def _get_valid_rep(self, items):
        if not items:
            return [random.choice(vals) for vals in self.independent_set]
        valid_items = [item for item in items if isinstance(item, list) and len(item) == self.D]
        return valid_items[0] if valid_items else [random.choice(vals) for vals in self.independent_set]

    def _better(self, a_reps, b_reps, budget_remaining):
        a_reps = [a_reps] if not isinstance(a_reps, list) else a_reps
        b_reps = [b_reps] if not isinstance(b_reps, list) else b_reps

        valid_a = [rep for rep in a_reps if isinstance(rep, list) and len(rep) == self.D]
        valid_b = [rep for rep in b_reps if isinstance(rep, list) and len(rep) == self.D]

        if not valid_a:
            valid_a = [self._get_valid_rep(self.initial_pop)]
        if not valid_b:
            valid_b = [self._get_valid_rep(self.initial_pop)]

        a_win = 0
        cost = 0
        for a in valid_a:
            for b in valid_b:
                if budget_remaining < 2:
                    return False, cost

                a_score, similar_a = get_objective_score_with_similarity(self.dict_search, a)
                similar_a_tuple = tuple(similar_a)
                if self.history_configs.get(similar_a_tuple) is None:
                    self.history_configs[similar_a_tuple] = a_score
                    self.xs.append(similar_a)
                    self.results.append(a_score)
                    cost += 1
                    budget_remaining -= 1
                else:
                    a_score = self.history_configs[similar_a_tuple]

                b_score, similar_b = get_objective_score_with_similarity(self.dict_search, b)
                similar_b_tuple = tuple(similar_b)
                if self.history_configs.get(similar_b_tuple) is None:
                    self.history_configs[similar_b_tuple] = b_score
                    self.xs.append(similar_b)
                    self.results.append(b_score)
                    cost += 1
                    budget_remaining -= 1
                else:
                    b_score = self.history_configs[similar_b_tuple]

                if a_score is None or b_score is None:
                    raise ValueError()

                if a_score < b_score:
                    a_win += 1

        return a_win > (len(valid_a) * len(valid_b)) / 2, cost

    def _sway_recursive(self, items, budget_remaining):
        if len(items) < self.enough or budget_remaining <= 0:
            return items, 0

        west, east, west_items, east_items = self._split_items(items)
        a_reps, b_reps = west, east

        total_cost = 0
        a_better = False
        if budget_remaining >= 2:
            a_better, cost = self._better(a_reps, b_reps, budget_remaining)
            total_cost += cost
            budget_remaining -= cost

        keep_west, keep_east = True, True
        if a_better:
            keep_east = False
        else:
            if budget_remaining >= 2:
                b_better, cost = self._better(b_reps, a_reps, budget_remaining)
                total_cost += cost
                budget_remaining -= cost
                if b_better:
                    keep_west = False

        result = []
        if keep_west and budget_remaining > 0:
            west_res, cost = self._sway_recursive(west_items, budget_remaining)
            result.extend(west_res)
            total_cost += cost
            budget_remaining -= cost
        if keep_east and budget_remaining > 0:
            east_res, cost = self._sway_recursive(east_items, budget_remaining)
            result.extend(east_res)
            total_cost += cost
            budget_remaining -= cost
        return result, total_cost

    def run_optimizers(self, budget=200):
        random.seed(self.seed)
        np.random.seed(self.seed)

        samples = self.file.all_set[:10000] if len(self.file.all_set) > 10000 else self.file.all_set
        random.shuffle(samples)
        for sample in samples:
            config = sample.decision
            config_tuple = tuple(config)
            if config_tuple not in [tuple(c) for c in self.initial_pop]:
                self.initial_pop.append(config)
        self.enough = max(1, int(np.sqrt(len(self.initial_pop))))
        
        raw_candidates, cluster_cost = self._sway_recursive(self.initial_pop, budget)
        remaining_budget = budget - cluster_cost
        
        best_result = min(self.results) if self.results else float('inf')
        best_config = self.xs[self.results.index(best_result)] if self.results else None
        consecutive_no_improve = 0
        step = len(self.results)

        for config in raw_candidates:
            if remaining_budget <= 0:
                break
            
            current_score, similar_config = get_objective_score_with_similarity(self.dict_search, config)
            similar_config_tuple = tuple(similar_config)

            if similar_config_tuple in self.history_configs:
                consecutive_no_improve += 1
                continue

            self.history_configs[similar_config_tuple] = current_score
            self.xs.append(similar_config)
            self.results.append(current_score)
            step += 1
            remaining_budget -= 1

            if current_score < best_result:
                best_result = current_score
                best_config = similar_config
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            if consecutive_no_improve >= self.maxlives:
                break

        total_used = budget - remaining_budget
        best_loop = self.results.index(best_result) + 1 if self.results else 0
        return (self.xs, self.results, step, best_result, best_loop, total_used)


def run_optimizers(file, budget=200, seed=0, maxlives=100):
    optimizer = SWAYOptimizer(file, seed=seed, maxlives=maxlives)
    return optimizer.run_optimizers(budget=budget)