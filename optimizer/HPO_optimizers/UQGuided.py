import random
import numpy as np
from scipy.stats import norm
from optimizer.util.QueryProblem import get_objective_score_with_similarity


class UQOptimizer:
    def __init__(self, file, seed=0, maxlives=100):
        self.file = file
        self.seed = seed
        self.independent_set = file.independent_set
        self.dict_search = file.dict_search
        self.best_result = float('inf')
        self.best_config = None
        self.best_loop = 0
        self.history = []
        self.used_budget = 0
        self.gamma = 0.1
        self.tau = 0.9
        self.evaluated_configs = set()
        
        self.consecutive_no_improve = 0
        self.loop = 0
        self.maxlives = maxlives

        random.seed(seed)
        np.random.seed(seed)

    def _estimate_uncertainty(self, config_history):
        losses = [h[2] for h in config_history]
        t = len(losses)
        if t < 2:
            return (losses[-1], 1.0)

        k_list = np.array([[1 / i ** 0.5, 1 / i] for i in range(1, t + 1)])
        A = np.hstack([np.ones((t, 1)), k_list])
        W = np.diag([1 / (i ** 0.5) for i in range(1, t + 1)])
        y = np.array(losses).reshape(-1, 1)
        beta = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y

        mu = beta[0, 0]
        sigma = np.var(y - A @ beta)
        return float(mu), float(sigma)

    def _confidence_curve(self, candidates):
        n = len(candidates)
        P = [0.0] * n

        for k in range(n):
            top_k = candidates[:k + 1]
            joint_prob = 1.0
            for i in range(k + 1):
                mu_i, sigma_i = top_k[i]['dist']
                for j in range(n):
                    if j > k:
                        mu_j, sigma_j = candidates[j]['dist']
                        prob = norm.cdf((mu_j - mu_i) / np.sqrt(sigma_i ** 2 + sigma_j ** 2))
                        joint_prob *= prob
            P[k] = joint_prob

        return P

    def _select_candidates(self, candidates, budget_remaining):
        if not candidates:
            return []

        candidates_sorted = sorted(candidates, key=lambda x: x['current_score'])
        for c in candidates_sorted:
            c['dist'] = self._estimate_uncertainty(c['history'])

        P = self._confidence_curve(candidates_sorted)
        k = next((i for i, p in enumerate(P) if p >= self.tau), len(candidates_sorted) - 1)
        max_keep = min(k + 1, budget_remaining)

        return candidates_sorted[:max_keep]

    def run_optimizers(self, budget=20):
        xs = []
        results = []

        candidates = []
        initial_size = min(5, budget)
        while (len(candidates) < initial_size 
               and self.used_budget < budget 
               and self.consecutive_no_improve < self.maxlives):
            self.loop += 1
            config = [random.choice(vals) for vals in self.independent_set]
            score, similar_config = get_objective_score_with_similarity(self.dict_search, config)
            similar_config_tuple = tuple(similar_config)

            if similar_config_tuple in self.evaluated_configs:
                self.consecutive_no_improve += 1
                continue

            self.used_budget += 1
            is_best = score < self.best_result

            if is_best:
                self.best_result = score
                self.best_config = similar_config
                self.best_loop = self.used_budget
                self.consecutive_no_improve = 0
            else:
                self.consecutive_no_improve += 1

            candidates.append({
                'config': config,
                'similar_config': similar_config,
                'current_score': score,
                'history': [(similar_config, None, score, self.loop)]
            })
            xs.append(similar_config)
            results.append(score)
            self.evaluated_configs.add(similar_config_tuple)

        while (self.used_budget < budget 
               and self.consecutive_no_improve < self.maxlives):
            remaining = budget - self.used_budget
            keep_candidates = self._select_candidates(candidates, remaining)
            if not keep_candidates:
                break

            new_candidates = []
            for c in keep_candidates:
                if self.used_budget >= budget or self.consecutive_no_improve >= self.maxlives:
                    break

                self.loop += 1
                new_config = c['config'].copy()
                for i in range(len(new_config)):
                    if random.random() < self.gamma:
                        new_config[i] = random.choice(self.independent_set[i])
                
                score, similar_config = get_objective_score_with_similarity(self.dict_search, new_config)
                similar_config_tuple = tuple(similar_config)

                if similar_config_tuple in self.evaluated_configs:
                    self.consecutive_no_improve += 1
                    continue

                self.used_budget += 1
                is_best = score < self.best_result

                if is_best:
                    self.best_result = score
                    self.best_config = similar_config
                    self.best_loop = self.used_budget
                    self.consecutive_no_improve = 0
                else:
                    self.consecutive_no_improve += 1

                new_candidates.append({
                    'config': new_config,
                    'similar_config': similar_config,
                    'current_score': score,
                    'history': [(similar_config, None, score, self.loop)]
                })
                xs.append(similar_config)
                results.append(score)
                self.evaluated_configs.add(similar_config_tuple)

            candidates = keep_candidates + new_candidates

        return (
            xs, results, range(1, len(results) + 1),
            self.best_result, self.best_loop, self.used_budget
        )


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    optimizer = UQOptimizer(file, seed=seed, maxlives=maxlives)
    return optimizer.run_optimizers(budget=budget)