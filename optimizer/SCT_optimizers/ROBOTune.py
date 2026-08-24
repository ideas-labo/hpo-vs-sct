import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from doepy import build
from optimizer.util.QueryProblem import get_objective_score_with_similarity


class ROBOTune:
    def __init__(self, file, seed=0):
        self.file = file
        self.seed = seed
        self.independent_set = file.independent_set
        self.dict_search = file.dict_search
        self.D = len(self.independent_set)

        self.param_cache = {}
        self.config_buffer = {}
        self.history_original_configs = dict()
        self.history_similar_configs = set()

        self.rf = RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            oob_score=True
        )
        self.gp_kernel = Matern(nu=2.5, length_scale_bounds=(1e-5, 1e6)) + WhiteKernel(
            noise_level=1.0, noise_level_bounds=(1e-10, 1000.0)
        )
        self.hedge_weights = {'ei': 1.0, 'pi': 1.0, 'lcb': 1.0}
        self.cumulative_gains = {'ei': 0.0, 'pi': 0.0, 'lcb': 0.0}

    def lhs_sampling(self, param_indices, num_samples):
        param_ranges = [self.independent_set[i] for i in param_indices]
        lhs_df = build.lhs(
            {f'p{i}': [0, 1] for i in range(len(param_indices))},
            num_samples=num_samples,
        )
        samples = []
        for _, row in lhs_df.iterrows():
            sample = []
            for i, idx in enumerate(param_indices):
                range_len = len(param_ranges[i])
                scaled = int(row[f'p{i}'] * range_len) % range_len
                sample.append(param_ranges[i][scaled])
            samples.append(sample)
        return samples

    def select_high_impact_params(self, workload_id, samples, scores):
        X = np.array([
            [self.independent_set[i].index(v) for i, v in enumerate(sample)]
            for sample in samples
        ])
        y = np.array(scores)
        self.rf.fit(X, y)

        oob_base = self.rf.oob_score_
        importances = []
        for i in range(X.shape[1]):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            oob_perm = self.rf.oob_score_
            importances.append(oob_base - oob_perm)

        threshold = 0.05 * max(importances) if importances else 0
        high_impact = [i for i, imp in enumerate(importances) if imp >= threshold]
        self.param_cache[workload_id] = high_impact
        return high_impact

    def acquisition_functions(self, X, X_train, y_train):
        gp = GaussianProcessRegressor(
            kernel=self.gp_kernel,
            n_restarts_optimizer=10,
            random_state=self.seed
        )
        gp.fit(X_train, y_train)
        mu, std = gp.predict(X, return_std=True)
        best_y = np.min(y_train)
        xi = 0.01
        kappa = 1.96

        imp = best_y - mu - xi
        Z = imp / (std + 1e-9)
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        pi = norm.cdf((best_y - mu - xi) / (std + 1e-9))
        lcb = mu - kappa * std

        return {'ei': ei, 'pi': pi, 'lcb': lcb}

    def hedge_selection(self, af_values):
        total_weight = sum(self.hedge_weights.values()) + 1e-10
        probs = {k: v / total_weight for k, v in self.hedge_weights.items()}
        probs = np.array(list(probs.values()))
        probs = probs / probs.sum()

        selected = np.random.choice(
            list(af_values.keys()),
            p=probs
        )
        return selected, af_values[selected]

    def update_hedge(self, selected_af, gain):
        self.cumulative_gains[selected_af] = max(-1.0, min(1.0, gain))
        learning_rate = 0.1
        for af in self.hedge_weights:
            self.hedge_weights[af] = max(1e-5,
                                         self.hedge_weights[af] * np.exp(learning_rate * self.cumulative_gains[af])
                                         )

    def run_optimizers(self, budget=20, workload_id="default", maxlives=100):
        xs = []
        results = []
        best_result = float('inf')
        best_config = None
        best_loop = 0
        consecutive_no_improve = 0
        loop = 0
        step = 0

        if workload_id in self.param_cache:
            high_impact = self.param_cache[workload_id]
            initial_samples = 20
            prev_configs = self.config_buffer.get(workload_id, [])
            num_prev = min(4, len(prev_configs))
        else:
            high_impact = list(range(self.D))
            initial_samples = 100
            prev_configs = self.config_buffer.get(workload_id, [])
            num_prev = 0

        initial_configs = []
        initial_configs.extend(prev_configs[:num_prev])
        lhs_samples = self.lhs_sampling(
            param_indices=high_impact,
            num_samples=initial_samples - num_prev
        )
        initial_configs.extend(lhs_samples)

        idx = 0
        while step < min(initial_samples, budget) and consecutive_no_improve < maxlives:
            if idx >= len(initial_configs):
                break
            current_original_config = initial_configs[idx]
            idx += 1
            original_config_tuple = tuple(current_original_config)
            loop += 1

            current_score, similar_config = get_objective_score_with_similarity(self.dict_search, current_original_config)
            similar_config_tuple = tuple(similar_config)

            if similar_config_tuple in self.history_similar_configs:
                consecutive_no_improve += 1
                continue

            self.history_original_configs[original_config_tuple] = current_score
            self.history_similar_configs.add(similar_config_tuple)
            xs.append(similar_config)
            results.append(current_score)
            step += 1

            if current_score < best_result:
                best_result = current_score
                best_config = similar_config
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

        if workload_id not in self.param_cache and step >= 100:
            X_rf = np.array([
                [self.independent_set[i].index(v) for i, v in enumerate(config)]
                for config in xs[:100]
            ])
            y_rf = np.array(results[:100])
            high_impact = self.select_high_impact_params(
                workload_id=workload_id,
                samples=xs[:100],
                scores=results[:100]
            )

        while step < budget and consecutive_no_improve < maxlives:
            X_train = np.array([
                [self.independent_set[i].index(v) for i, v in enumerate(config)]
                for config in xs
            ])
            y_train = np.array(results)

            candidates_original = self.lhs_sampling(high_impact, num_samples=100)
            X_candidates = np.array([
                [self.independent_set[i].index(v) for i, v in enumerate(conf)]
                for conf in candidates_original
            ])

            af_values = self.acquisition_functions(X_candidates, X_train, y_train)
            selected_af, af_scores = self.hedge_selection(af_values)
            best_candidate_idx = np.argmax(af_scores)
            next_original_config = candidates_original[best_candidate_idx]
            next_original_tuple = tuple(next_original_config)
            loop += 1

            next_score, similar_config = get_objective_score_with_similarity(self.dict_search, next_original_config)
            similar_config_tuple = tuple(similar_config)

            if similar_config_tuple in self.history_similar_configs:
                consecutive_no_improve += 1
                continue

            current_best = min(y_train) if len(y_train) > 0 else float('inf')
            gain = current_best - next_score
            self.update_hedge(selected_af, gain)

            self.history_original_configs[next_original_tuple] = next_score
            self.history_similar_configs.add(similar_config_tuple)
            xs.append(similar_config)
            results.append(next_score)
            step += 1

            if next_score < best_result:
                best_result = next_score
                best_config = similar_config
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

        if xs and results:
            sorted_pairs = sorted(zip(results, xs), key=lambda x: x[0])
            sorted_similar_configs = [x for _, x in sorted_pairs]
            self.config_buffer[workload_id] = sorted_similar_configs[:4]

        return (
            xs[:budget],
            results[:budget],
            range(1, len(results[:budget]) + 1),
            best_result,
            best_loop,
            min(step, budget)
        )


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    optimizer = ROBOTune(file, seed=seed)
    workload_id = file.name.split('/')[-1] if hasattr(file, 'name') else "default"
    return optimizer.run_optimizers(budget=budget, workload_id=workload_id, maxlives=maxlives)