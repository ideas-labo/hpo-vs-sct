import math
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from optimizer.util.QueryProblem import get_objective_score_with_similarity

class SMACOptimizer:
    def __init__(self, param_space, n_trees=10, ei_weight=0.5):
        self.param_space = param_space
        self.n_trees = n_trees
        self.ei_weight = ei_weight
        self.model = None
        self.X = []
        self.y = []
        self.incumbent = None
        self.incumbent_score = float('inf')

    def _convert_config_to_features(self, config):
        features = []
        for i, val in enumerate(config):
            param_values = self.param_space[i]
            features.append(param_values.index(val))
        return features

    def fit_model(self):
        if len(self.X) < 2:
            return

        y_values = np.array(self.y)
        min_value = -1e-5
        y_non_negative = np.maximum(y_values, min_value) - min_value + 1e-5
        scale_factor = 1e5
        y_scaled = y_non_negative * scale_factor
        y_log = np.log1p(y_scaled)

        self.model = RandomForestRegressor(
            n_estimators=self.n_trees,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X, y_log)

    def _expected_improvement(self, config):
        if self.model is None:
            return random.random()
        features = self._convert_config_to_features(config)
        y_preds = [tree.predict([features])[0] for tree in self.model.estimators_]
        mu = np.mean(y_preds)
        sigma = np.std(y_preds)

        incumbent_score_non_negative = max(self.incumbent_score, -1e-5) - (-1e-5) + 1e-5
        incumbent_score_scaled = incumbent_score_non_negative * 1e5
        f_min = np.log1p(incumbent_score_scaled)

        if sigma < 1e-6:
            return 0.0
        z = (f_min - mu) / sigma
        cdf = 0.5 * (1 + math.erf(z / np.sqrt(2)))
        pdf = np.exp(-0.5 * z **2) / np.sqrt(2 * np.pi)
        ei = (f_min - mu) * cdf + sigma * pdf
        return ei

    def select_next_config(self, num_candidates=1000):
        num_candidates = int(num_candidates) if isinstance(num_candidates, (int, float)) else 1000
        candidates = []
        for _ in range(num_candidates):
            config = [random.choice(vals) for vals in self.param_space]
            candidates.append(config)

        candidates.sort(key=lambda x: self._expected_improvement(x), reverse=True)
        return candidates[0]

    def intensify(self, config, dict_search, history_configs):
        score, similar_config = get_objective_score_with_similarity(dict_search, config)
        similar_config_tuple = tuple(similar_config)

        if similar_config_tuple not in history_configs and score < self.incumbent_score:
            self.incumbent = similar_config
            self.incumbent_score = score
        return score, similar_config

def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)

    param_space = file.independent_set
    dict_search = file.dict_search

    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []
    history_configs = set()
    loop = 0
    budget_used = 0
    consecutive_no_improve = 0
    num_candidates = 1000
    optimizer = SMACOptimizer(param_space)

    initial_config = [random.choice(vals) for vals in param_space]
    initial_score, initial_similar = get_objective_score_with_similarity(dict_search, initial_config)
    initial_similar_tuple = tuple(initial_similar)

    loop += 1
    budget_used += 1
    history_configs.add(initial_similar_tuple)
    xs.append(initial_similar)
    results.append(initial_score)
    optimizer.X.append(optimizer._convert_config_to_features(initial_similar))
    optimizer.y.append(initial_score)

    optimizer.incumbent = initial_similar
    optimizer.incumbent_score = initial_score
    best_result = initial_score
    best_config = initial_similar
    best_loop = budget_used

    while budget_used < budget and consecutive_no_improve < maxlives:
        loop += 1

        optimizer.fit_model()
        next_config = optimizer.select_next_config(int(num_candidates))
        next_score, next_similar = get_objective_score_with_similarity(dict_search, next_config)
        next_similar_tuple = tuple(next_similar)

        if next_similar_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        budget_used += 1
        history_configs.add(next_similar_tuple)
        xs.append(next_similar)
        results.append(next_score)
        optimizer.X.append(optimizer._convert_config_to_features(next_similar))
        optimizer.y.append(next_score)

        optimizer.intensify(next_config, dict_search, history_configs)

        if next_score < best_result:
            best_result = next_score
            best_config = next_similar
            best_loop = budget_used
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    final_round = len(xs)
    return xs, results, range(1, final_round + 1), best_result, best_loop, final_round