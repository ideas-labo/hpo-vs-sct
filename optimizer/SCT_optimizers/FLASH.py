import random
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from optimizer.util.QueryProblem import get_objective_score_with_similarity

class FLASHOptimizer:
    def __init__(self, file, seed=0):
        self.file = file
        self.seed = seed
        self.independent_set = file.independent_set
        self.dict_search = file.dict_search
        self.D = len(self.independent_set)
        
        self.history_original = dict()
        self.history_similar = set()
        self.score_cache = {}

        self.init_size = 30
        self.candidate_pool_size = 100
        self.cart_max_depth = None

        self.param_value_to_idx = [
            {v: idx for idx, v in enumerate(param_range)}
            for param_range in self.independent_set
        ]

    def _build_cart_model(self, X, y):
        cart = DecisionTreeRegressor(
            max_depth=self.cart_max_depth,
            random_state=self.seed
        )
        cart.fit(X, y)
        return cart

    def _select_best_candidate(self, candidates, model):
        X_candidate = self._encode_configs(candidates)
        preds = model.predict(X_candidate)
        return candidates[np.argmin(preds)]

    def _encode_config(self, config):
        return [self.param_value_to_idx[i][val] for i, val in enumerate(config)]

    def _encode_configs(self, configs):
        return np.array([self._encode_config(c) for c in configs])

    def run_optimizers(self, budget=20, maxlives=100):
        random.seed(self.seed)
        np.random.seed(self.seed)

        xs = []
        results = []
        best_result = None
        best_config = None
        best_loop = 0
        loop = 0
        step = 0
        consecutive_no_improve = 0

        while step < min(self.init_size, budget):
            original_config = [random.choice(vals) for vals in self.independent_set]
            original_tuple = tuple(original_config)
            loop += 1

            score, similar_config = get_objective_score_with_similarity(self.dict_search, original_config)
            similar_tuple = tuple(similar_config)
            
            if similar_tuple in self.history_similar:
                consecutive_no_improve += 1
                if consecutive_no_improve >= maxlives:
                    return self._format_result(xs, results, best_result, best_loop, step)
                continue

            self.history_original[original_tuple] = score
            self.history_similar.add(similar_tuple)
            self.score_cache[original_tuple] = score

            xs.append(similar_config)
            results.append(score)
            step += 1

            if best_result is None or score < best_result:
                best_result = score
                best_config = similar_config
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            if consecutive_no_improve >= maxlives:
                return self._format_result(xs, results, best_result, best_loop, step)

        remaining_budget = budget - step
        if remaining_budget <= 0:
            return self._format_result(xs, results, best_result, best_loop, step)

        while remaining_budget > 0 and consecutive_no_improve < maxlives:
            X = self._encode_configs(xs)
            y = np.array(results)
            model = self._build_cart_model(X, y)

            candidates = []
            while len(candidates) < self.candidate_pool_size:
                config = [random.choice(vals) for vals in self.independent_set]
                config_tuple = tuple(config)
                if config_tuple not in self.history_similar:
                    candidates.append(config)

            next_original = self._select_best_candidate(candidates, model)
            next_original_tuple = tuple(next_original)
            loop += 1

            score, similar_config = get_objective_score_with_similarity(self.dict_search, next_original)
            similar_tuple = tuple(similar_config)

            if similar_tuple in self.history_similar:
                consecutive_no_improve += 1
                continue

            self.history_original[next_original_tuple] = score
            self.history_similar.add(similar_tuple)
            self.score_cache[next_original_tuple] = score

            xs.append(similar_config)
            results.append(score)
            step += 1
            remaining_budget -= 1

            if score < best_result:
                best_result = score
                best_config = similar_config
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

        return self._format_result(xs, results, best_result, best_loop, step)

    def _format_result(self, xs, results, best_result, best_loop, used_budget):
        return (
            xs[:used_budget],
            results[:used_budget],
            range(1, used_budget + 1),
            best_result,
            best_loop,
            used_budget
        )

def run_optimizers(file, budget=20, seed=0, maxlives=100):
    optimizer = FLASHOptimizer(file, seed=seed)
    return optimizer.run_optimizers(budget=budget, maxlives=maxlives)