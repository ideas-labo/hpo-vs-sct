import random
import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel
from optimizer.util.QueryProblem import get_objective_score_with_similarity

class IraceOptimizer:
    def __init__(self, file, seed=0, maxlives=100):
        self.file = file
        self.seed = seed
        self.maxlives = maxlives
        self.independent_set = file.independent_set
        self.dict_search = file.dict_search
        self.D = len(self.independent_set)

        self.N_iter = max(2, int(np.floor(2 + np.log2(self.D))))
        self.T_first = 5
        self.T_each = 1
        self.N_min = 2
        self.alpha = 0.05
        self.test_type = "friedman"
        self.restart_threshold = 1e-4

        self.elite_configs = []
        self.elite_generated = []
        self.sampling_distributions = self._init_sampling_distributions()
        self.history_configs = {}
        self.best_result = float('inf')
        self.best_config = None
        self.consecutive_no_improve = 0
        self.loop = 0

    def _init_sampling_distributions(self):
        distributions = []
        for param_values in self.independent_set:
            n = len(param_values)
            distributions.append({
                'type': 'enumerate',
                'probs': [1 / n for _ in range(n)],
                'values': param_values
            })
        return distributions

    def _sample_config(self):
        config = []
        for i in range(self.D):
            dist = self.sampling_distributions[i]
            config.append(np.random.choice(dist['values'], p=dist['probs']))
        return config

    def _update_distributions(self, elite_generated):
        if not elite_generated:
            return
        
        for i in range(self.D):
            dist = self.sampling_distributions[i]
            elite_values = [cfg[i] for cfg in elite_generated]
            counts = [elite_values.count(v) for v in dist['values']]
            total = sum(counts)
            if total == 0:
                continue

            new_probs = [
                (1 - (len(elite_generated) - 1) / self.N_iter) * p
                + (len(elite_generated) - 1) / self.N_iter * (count / total)
                for p, count in zip(dist['probs'], counts)
            ]
            new_probs_np = np.array(new_probs)
            new_probs_np /= new_probs_np.sum()
            self.sampling_distributions[i]['probs'] = new_probs_np.tolist()

    def _statistical_test(self, configs, effectivenesss, generated_configs):
        if len(configs) <= self.N_min:
            return configs, generated_configs

        max_len = max(len(p) for p in effectivenesss)
        padded_effectivenesss = [p + [np.nan] * (max_len - len(p)) for p in effectivenesss]
        perf_matrix = np.array(padded_effectivenesss).T

        if self.test_type == 'friedman':
            try:
                valid_cols = ~np.isnan(perf_matrix).any(axis=0)
                valid_perf_matrix = perf_matrix[:, valid_cols]
                stat, p_value = friedmanchisquare(*valid_perf_matrix)
            except ValueError:
                return configs, generated_configs

            if p_value >= self.alpha:
                return configs, generated_configs

            avg_ranks = np.mean([np.argsort(np.argsort(perf)) for perf in valid_perf_matrix], axis=0)
            threshold = np.percentile(avg_ranks, 80)
            valid_configs = [cfg for cfg, valid in zip(configs, valid_cols) if valid]
            valid_generated = [cfg for cfg, valid in zip(generated_configs, valid_cols) if valid]
            survivors = [cfg for cfg, rank in zip(valid_configs, avg_ranks) if rank <= threshold]
            survivors_generated = [cfg for cfg, rank in zip(valid_generated, avg_ranks) if rank <= threshold]
            return survivors, survivors_generated
        else:
            best_idx = np.argmin([np.nanmean(p) for p in padded_effectivenesss])
            survivors = [configs[best_idx]]
            survivors_generated = [generated_configs[best_idx]]
            for i in range(len(configs)):
                if i == best_idx:
                    continue
                if len(padded_effectivenesss[i]) < self.T_first:
                    survivors.append(configs[i])
                    survivors_generated.append(generated_configs[i])
                    continue
                valid_mask = ~np.isnan(padded_effectivenesss[best_idx]) & ~np.isnan(padded_effectivenesss[i])
                valid_best = np.array(padded_effectivenesss[best_idx])[valid_mask]
                valid_other = np.array(padded_effectivenesss[i])[valid_mask]
                if len(valid_best) < 2 or len(valid_other) < 2:
                    survivors.append(configs[i])
                    survivors_generated.append(generated_configs[i])
                    continue
                _, p = ttest_rel(valid_best, valid_other)
                if p >= self.alpha:
                    survivors.append(configs[i])
                    survivors_generated.append(generated_configs[i])
            return survivors, survivors_generated

    def _check_restart(self, candidates):
        if len(candidates) < 2:
            return False

        distances = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                max_dist = 0
                for k in range(self.D):
                    dist = 0 if candidates[i][k] == candidates[j][k] else 1
                    max_dist = max(max_dist, dist)
                distances.append(max_dist)

        if np.mean(distances) < self.restart_threshold:
            for i in range(self.D):
                n = len(self.sampling_distributions[i]['values'])
                self.sampling_distributions[i]['probs'] = [1 / n for _ in range(n)]
            return True
        return False

    def run_optimizers(self, budget=200):
        random.seed(self.seed)
        np.random.seed(self.seed)

        xs = []
        results = []
        generated_configs = []
        remaining_budget = budget
        iter_count = 1

        self.consecutive_no_improve = 0
        self.history_configs = {}
        self.best_result = float('inf')
        self.best_config = None
        self.loop = 0

        while remaining_budget > 0 and self.consecutive_no_improve < self.maxlives:
            if self.consecutive_no_improve >= self.maxlives:
                break

            if iter_count >= self.N_iter:
                self.N_iter += 1

            new_candidates = [self._sample_config() for _ in range(min(10, remaining_budget))]
            self._check_restart(new_candidates)

            candidates = new_candidates + self.elite_generated
            
            for cfg in candidates:
                if remaining_budget <= 0 or self.consecutive_no_improve >= self.maxlives:
                    break
                    
                self.loop += 1
                current_score, similar_config = get_objective_score_with_similarity(self.dict_search, cfg)
                similar_tuple = tuple(similar_config)
                
                if similar_tuple in self.history_configs:
                    self.consecutive_no_improve += 1
                    continue
                else:
                    self.history_configs[similar_tuple] = current_score
                    xs.append(similar_config)
                    generated_configs.append(cfg)
                    results.append(current_score)
                    remaining_budget -= 1
                    
                    if current_score < self.best_result:
                        self.best_result = current_score
                        self.best_config = similar_config
                        self.consecutive_no_improve = 0
                    else:
                        self.consecutive_no_improve += 1

            if xs:
                cfg_perf = list(zip(xs, results, generated_configs))
                cfg_perf.sort(key=lambda x: x[1])
                self.elite_configs = [cfg for cfg, _, _ in cfg_perf[:self.N_min]]
                self.elite_generated = [g for _, _, g in cfg_perf[:self.N_min]]
                self._update_distributions(self.elite_generated)

            iter_count += 1
            if iter_count > 1000:
                break

        best_loop = results.index(self.best_result) + 1 if results else 0
        return (xs, results, range(1, len(results) + 1),
                self.best_result, best_loop, len(xs))


def run_optimizers(file, budget=200, seed=0, maxlives=100):
    optimizer = IraceOptimizer(file, seed=seed, maxlives=maxlives)
    return optimizer.run_optimizers(budget=budget)