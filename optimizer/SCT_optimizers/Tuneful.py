import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from optimizer.util.QueryProblem import get_objective_score_with_similarity


class TunefulOptimizer:
    def __init__(self, file, seed=0):
        self.file = file
        self.seed = seed
        self.independent_set = file.independent_set
        self.dict_search = file.dict_search
        self.D = len(self.independent_set)
        self.history_configs = {}
        self.generated_configs = set()
        self.score_cache = {}

        self.sa_alpha = 0.6
        self.sa_samples_per_round = 10
        self.sa_max_rounds = 2

        self.default_values = [param_range[0] for param_range in self.independent_set]
        self.param_value_to_idx = [
            {v: idx for idx, v in enumerate(param_range)}
            for param_range in self.independent_set
        ]

        self.consecutive_no_improve = 0
        self.maxlives = None

    def _get_parameter_importance(self, samples, scores, significant_params):
        X = np.array([
            [self.param_value_to_idx[param_idx][sample[param_idx]]
             for param_idx in significant_params]
            for sample in samples
        ])
        y = np.array(scores)
        rf = RandomForestRegressor(n_estimators=100, random_state=self.seed)
        rf.fit(X, y)
        return rf.feature_importances_

    def _significant_analyzer(self):
        significant_params = list(range(self.D))
        current_samples = []
        current_scores = []
        best_score = float('inf')

        for round_idx in range(self.sa_max_rounds):
            num_samples = self.sa_samples_per_round
            sampler = Sobol(d=len(significant_params), scramble=True, seed=self.seed)
            samples = sampler.random_base2(m=int(np.ceil(np.log2(num_samples))))[:num_samples]

            for sample in samples:
                full_config = self.default_values.copy()
                for i, param_idx in enumerate(significant_params):
                    param_range = self.independent_set[param_idx]
                    value_idx = int(sample[i] * len(param_range))
                    value_idx = min(value_idx, len(param_range) - 1)
                    full_config[param_idx] = param_range[value_idx]
                
                full_config_tuple = tuple(full_config)
                if full_config_tuple in self.generated_configs:
                    self.consecutive_no_improve += 1
                    if self.consecutive_no_improve >= self.maxlives:
                        return significant_params, current_samples, current_scores
                    continue

                self.generated_configs.add(full_config_tuple)
                score, similar_config = get_objective_score_with_similarity(self.dict_search, full_config)
                similar_tuple = tuple(similar_config)

                if similar_tuple in self.history_configs:
                    self.consecutive_no_improve += 1
                    if self.consecutive_no_improve >= self.maxlives:
                        return significant_params, current_samples, current_scores
                    continue

                self.history_configs[similar_tuple] = score
                current_samples.append(similar_config)
                current_scores.append(score)

                if score < best_score:
                    best_score = score
                    self.consecutive_no_improve = 0
                else:
                    self.consecutive_no_improve += 1

                if self.consecutive_no_improve >= self.maxlives:
                    return significant_params, current_samples, current_scores

                if len(current_samples) % num_samples == 0:
                    break

            importances = self._get_parameter_importance(
                current_samples[-num_samples:],
                current_scores[-num_samples:],
                significant_params
            )
            num_keep = max(5, int(len(significant_params) * self.sa_alpha))
            significant_indices = np.argsort(importances)[-num_keep:]
            significant_params = [significant_params[i] for i in significant_indices]

        return significant_params, current_samples, current_scores

    def _gp_optimizer(self, significant_params, init_samples, init_scores, budget):
        param_ranges = [self.independent_set[i] for i in significant_params]
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=self.seed
        )

        X = np.array([
            [self.param_value_to_idx[significant_params[i]][sample[significant_params[i]]]
            for i in range(len(significant_params))]
            for sample in init_samples
        ])
        y = np.array(init_scores)
        best_score = np.min(y) if len(y) > 0 else float('inf')
        best_score_prev = best_score

        if len(X) > 0:
            gp.fit(X, y)

        samples = init_samples.copy()
        scores = init_scores.copy()
        used_budget = len(samples)

        while used_budget < budget:
            generated_candidates = []
            for _ in range(100):
                full_config = self.default_values.copy()
                for i, param_idx in enumerate(significant_params):
                    full_config[param_idx] = random.choice(param_ranges[i])
                generated_candidates.append(full_config)

            X_candidate = np.array([
                [self.param_value_to_idx[significant_params[i]][c[significant_params[i]]]
                for i in range(len(significant_params))]
                for c in generated_candidates
            ])
            mu, std = gp.predict(X_candidate, return_std=True)
            best_y = np.min(y) if len(y) > 0 else float('inf')
            ei = (best_y - mu) * norm.cdf((best_y - mu) / (std + 1e-9)) + \
                 std * norm.pdf((best_y - mu) / (std + 1e-9))
            best_idx = np.argmax(ei)
            next_generated_config = generated_candidates[best_idx]
            
            next_config_tuple = tuple(next_generated_config)
            if next_config_tuple in self.generated_configs:
                self.consecutive_no_improve += 1
                if self.consecutive_no_improve >= self.maxlives:
                    break
                continue
            
            self.generated_configs.add(next_config_tuple)
            next_score, next_similar_config = get_objective_score_with_similarity(self.dict_search, next_generated_config)
            next_similar_tuple = tuple(next_similar_config)

            if next_similar_tuple in self.history_configs:
                self.consecutive_no_improve += 1
                if self.consecutive_no_improve >= self.maxlives:
                    break
                continue

            self.history_configs[next_similar_tuple] = next_score
            samples.append(next_similar_config)
            scores.append(next_score)
            used_budget += 1

            X = np.vstack([X, [[
                self.param_value_to_idx[significant_params[i]][next_similar_config[significant_params[i]]]
                for i in range(len(significant_params))
            ]]])
            y = np.append(y, next_score)
            gp.fit(X, y)

            current_best = np.min(y)
            if current_best < best_score:
                best_score = current_best
                self.consecutive_no_improve = 0
            else:
                self.consecutive_no_improve += 1

            best_score_prev = best_score

            if self.consecutive_no_improve >= self.maxlives:
                break

        return samples, scores

    def run_optimizers(self, budget=20, maxlives=100):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.maxlives = maxlives
        self.consecutive_no_improve = 0
        self.generated_configs.clear()
        xs = []
        results = []
        best_result = float('inf')
        best_loop = 0

        significant_params, init_samples, init_scores = self._significant_analyzer()
        total_samples, total_scores = self._gp_optimizer(
            significant_params,
            init_samples,
            init_scores,
            budget
        )

        for i, (config, score) in enumerate(zip(total_samples, total_scores)):
            if i >= budget:
                break
            config_tuple = tuple(config)
            if config_tuple not in self.history_configs:
                continue
            xs.append(config)
            results.append(score)
            if score < best_result:
                best_result = score
                best_loop = len(xs)

        used_budget = min(len(xs), budget)
        return (
            xs[:used_budget],
            results[:used_budget],
            range(1, used_budget + 1),
            best_result,
            best_loop,
            used_budget
        )


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    optimizer = TunefulOptimizer(file, seed=seed)
    return optimizer.run_optimizers(budget=budget, maxlives=maxlives)