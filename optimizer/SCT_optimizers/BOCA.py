import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from itertools import product
from optimizer.util.QueryProblem import get_objective_score_with_similarity


def run_optimizers(file, budget=1000, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)
    max_regen_attempts=1000
    independent_set = file.independent_set
    dict_search = file.dict_search
    D = len(independent_set)

    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_configs = set()
    consecutive_no_improve = 0
    loop = 0
    used_budget = 0

    initial_samples = min(5, budget)
    step = 0
    while step < initial_samples and used_budget < budget and consecutive_no_improve < maxlives:
        loop += 1
        generated_config = [random.choice(values) for values in independent_set]
        current_score, similar_config = get_objective_score_with_similarity(dict_search, generated_config)
        similar_config_tuple = tuple(similar_config)

        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        valid = all(val in independent_set[i] for i, val in enumerate(generated_config))
        if not valid:
            consecutive_no_improve += 1
            continue

        history_configs.add(similar_config_tuple)
        xs.append(similar_config)
        results.append(current_score)
        used_budget += 1
        step += 1

        if current_score < best_result:
            best_result = current_score
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    if consecutive_no_improve >= maxlives or used_budget >= budget:
        return xs, results, range(1, loop + 1), best_result, best_loop, used_budget

    C1 = 20
    offset = 20
    scale = 0.5
    decay = 10
    K = min(D-1,8)
    max_values_per_impactful_param = 10

    while used_budget < budget and consecutive_no_improve < maxlives:
        loop += 1
        current_iter = used_budget - initial_samples + 1

        X_train = np.array([[independent_set[i].index(val) for i, val in enumerate(config)] for config in xs])
        y_train = np.array(results)
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1, bootstrap=True)
        model.fit(X_train, y_train)

        feature_importance = model.feature_importances_
        impactful_indices = np.argsort(feature_importance)[-K:][::-1]
        less_impactful_indices = [i for i in range(D) if i not in impactful_indices]

        exponent = -((current_iter - offset) / scale) **2 / (2 * decay**2)
        C_i = max(int(C1 * np.exp(exponent)), 1)

        impactful_values = []
        for idx in impactful_indices:
            param_values = independent_set[idx]
            if len(param_values) > max_values_per_impactful_param:
                impactful_values.append(random.sample(param_values, max_values_per_impactful_param))
            else:
                impactful_values.append(param_values)
        impactful_combinations = list(product(*impactful_values))
        candidates = []
        less_impactful_values = [independent_set[i] for i in less_impactful_indices]
        
        for imp_comb in impactful_combinations[:5]:
            for _ in range(C_i):
                less_comb = [random.choice(vals) for vals in less_impactful_values]
                candidate = [None] * D
                imp_idx, less_idx = 0, 0
                for i in range(D):
                    if i in impactful_indices:
                        candidate[i] = imp_comb[imp_idx]
                        imp_idx += 1
                    else:
                        candidate[i] = less_comb[less_idx]
                        less_idx += 1
                if all(val in independent_set[i] for i, val in enumerate(candidate)) and candidate not in candidates:
                    candidates.append(candidate)

        next_generated_config = None
        next_similar_config = None
        next_similar_tuple = None
        
        if not candidates:
            regen_attempts = 0
            next_generated_config = [random.choice(values) for values in independent_set]
            next_score, next_similar_config = get_objective_score_with_similarity(dict_search, next_generated_config)
            next_similar_tuple = tuple(next_similar_config)
            
            while next_similar_tuple in history_configs and regen_attempts < max_regen_attempts:
                regen_attempts += 1
                next_generated_config = [random.choice(values) for values in independent_set]
                next_score, next_similar_config = get_objective_score_with_similarity(dict_search, next_generated_config)
                next_similar_tuple = tuple(next_similar_config)
                
            if regen_attempts >= max_regen_attempts:
                consecutive_no_improve += 1
                continue
        else:
            X_candidates = []
            valid_candidates = []
            for config in candidates:
                try:
                    X_candidates.append([independent_set[i].index(val) for i, val in enumerate(config)])
                    valid_candidates.append(config)
                except ValueError:
                    continue
            X_candidates = np.array(X_candidates)

            y_pred = model.predict(X_candidates)
            all_preds = np.array([tree.predict(X_candidates) for tree in model.estimators_])
            y_std = np.std(all_preds, axis=0)

            best_y = np.min(y_train)
            imp = best_y - y_pred
            Z = imp / (y_std + 1e-9)
            ei = imp * norm.cdf(Z) + y_std * norm.pdf(Z)
            next_generated_config = valid_candidates[np.argmax(ei)]
            next_score, next_similar_config = get_objective_score_with_similarity(dict_search, next_generated_config)
            next_similar_tuple = tuple(next_similar_config)

        if next_similar_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        history_configs.add(next_similar_tuple)
        xs.append(next_similar_config)
        results.append(next_score)
        used_budget += 1

        if next_score < best_result:
            best_result = next_score
            best_loop = used_budget
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return xs, results, range(1, loop + 1), best_result, best_loop, used_budget