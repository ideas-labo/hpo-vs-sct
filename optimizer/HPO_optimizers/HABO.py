import numpy as np
from optimizer.util.QueryProblem import get_objective_score_with_similarity

class HABOOptimizer:
    def __init__(self, hyperparameters: dict, gamma: float = 0.1):
        self.hyperparameters = hyperparameters
        self.gamma = gamma
        self.super_arms = list(hyperparameters.keys())
        self.k = len(self.super_arms)

        self.super_weights = {arm: 1.0 for arm in self.super_arms}
        self.sub_weights = {
            super_arm: {sub_arm: 1.0 for sub_arm in hyperparameters[super_arm]}
            for super_arm in self.super_arms
        }

        self.current_config = {super_arm: None for super_arm in self.super_arms}

    def _select_super_arm(self) -> str:
        total_weight = sum(self.super_weights.values())
        probabilities = {
            arm: (1 - self.gamma) * (weight / total_weight) + self.gamma / self.k
            for arm, weight in self.super_weights.items()
        }
        arms = list(probabilities.keys())
        probs = list(probabilities.values())
        return np.random.choice(arms, p=probs)

    def _select_sub_arm(self, super_arm: str) -> any:
        sub_arms = self.hyperparameters[super_arm]
        sub_weights = self.sub_weights[super_arm]
        total_weight = sum(sub_weights.values())

        if total_weight == 0:
            return np.random.choice(sub_arms)
        
        if np.isinf(total_weight):
            valid_weights = {k: v for k, v in sub_weights.items() if not np.isinf(v)}
            if valid_weights:
                return max(valid_weights, key=valid_weights.get)
            else:
                return np.random.choice(sub_arms)

        probabilities = {sub_arm: weight / total_weight for sub_arm, weight in sub_weights.items()}
        sub_arms_list = list(probabilities.keys())
        sub_probs = list(probabilities.values())
        return np.random.choice(sub_arms_list, p=sub_probs)

    def generate_config(self) -> dict:
        super_arm = self._select_super_arm()
        sub_arm = self._select_sub_arm(super_arm)
        self.current_config[super_arm] = sub_arm

        for arm in self.super_arms:
            if self.current_config[arm] is None:
                self.current_config[arm] = self.hyperparameters[arm][0]
        
        return self.current_config.copy(), super_arm, sub_arm

    def update_weights(self, selected_super_arm: str, selected_sub_arm: any, reward: float) -> None:
        total_super_weight = sum(self.super_weights.values())
        super_prob = (1 - self.gamma) * (self.super_weights[selected_super_arm] / total_super_weight) + self.gamma / self.k
        exp_super = min(self.gamma * reward / (self.k * super_prob), 709)
        self.super_weights[selected_super_arm] *= np.exp(exp_super)

        total_sub_weight = sum(self.sub_weights[selected_super_arm].values())
        sub_prob = self.sub_weights[selected_super_arm][selected_sub_arm] / total_sub_weight
        exp_sub = min(self.gamma * reward / sub_prob, 709)
        self.sub_weights[selected_super_arm][selected_sub_arm] *= np.exp(exp_sub)

    def reset(self) -> None:
        self.super_weights = {arm: 1.0 for arm in self.super_arms}
        self.sub_weights = {
            super_arm: {sub_arm: 1.0 for sub_arm in self.hyperparameters[super_arm]}
            for super_arm in self.super_arms
        }
        self.current_config = {super_arm: None for super_arm in self.super_arms}


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    np.random.seed(seed)
    independent_set = file.independent_set
    dict_search = file.dict_search

    hyperparameters = {f'param_{i}': values for i, values in enumerate(independent_set)}
    optimizer = HABOOptimizer(hyperparameters)

    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []
    history_configs = dict()
    consecutive_no_improve = 0
    step = 0
    loop = 0

    while step < budget and consecutive_no_improve < maxlives:
        loop += 1

        current_config, selected_super_arm, selected_sub_arm = optimizer.generate_config()
        current_config_values = [current_config[f'param_{i}'] for i in range(len(independent_set))]
        
        current_score, similar_config = get_objective_score_with_similarity(dict_search, current_config_values)
        similar_config_tuple = tuple(similar_config)

        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        history_configs[similar_config_tuple] = current_score

        score_min = min([v for v in history_configs.values()] + [current_score])
        score_max = max([v for v in history_configs.values()] + [current_score])
        if score_max == score_min:
            reward = 0.5
        else:
            reward = (score_max - current_score) / (score_max - score_min)

        optimizer.update_weights(selected_super_arm, selected_sub_arm, reward)

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

    return xs, results, range(1, step + 1), best_result, best_loop, step