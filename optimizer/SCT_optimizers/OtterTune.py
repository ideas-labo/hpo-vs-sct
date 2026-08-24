import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

from optimizer.util.QueryProblem import get_objective_score_with_similarity


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    history_repo = {}

    random.seed(seed)
    np.random.seed(seed)

    independent_set = file.independent_set
    dict_search = file.dict_search
    D = len(independent_set)

    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_original = set()
    history_similar = set()
    loop = 0
    consecutive_no_improve = 0

    X_history = []
    y_history = []
    for _, (X, y) in history_repo.items():
        X_history.extend(X)
        y_history.extend(y)
    X_history = np.array(X_history) if X_history else np.array([])
    y_history = np.array(y_history) if y_history else np.array([])

    initial_samples = min(5, budget)
    step = 0

    if len(y_history) > 0:
        history_best_idx = np.argmin(y_history)
        history_best_config = X_history[history_best_idx].tolist()
        original_tuple = tuple(history_best_config)
        
        score, similar_config = get_objective_score_with_similarity(dict_search, history_best_config)
        similar_tuple = tuple(similar_config)
        
        if similar_tuple not in history_similar:
            history_original.add(original_tuple)
            history_similar.add(similar_tuple)
            xs.append(similar_config)
            results.append(score)
            
            if score < best_result:
                best_result = score
                best_loop = step + 1
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
            
            step += 1
            loop += 1
        else:
            loop += 1
            consecutive_no_improve += 1

    while step < initial_samples and consecutive_no_improve < maxlives:
        original_config = [random.choice(vals) for vals in independent_set]
        original_tuple = tuple(original_config)
        loop += 1

        score, similar_config = get_objective_score_with_similarity(dict_search, original_config)
        similar_tuple = tuple(similar_config)

        if similar_tuple in history_similar:
            consecutive_no_improve += 1
            continue

        history_original.add(original_tuple)
        history_similar.add(similar_tuple)
        xs.append(similar_config)
        results.append(score)

        if score < best_result:
            best_result = score
            best_loop = step + 1
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

        step += 1

    if step >= budget or consecutive_no_improve >= maxlives:
        used_budget = min(step, budget)
        return (xs[:used_budget], results[:used_budget], 
                range(1, used_budget + 1), best_result, best_loop, used_budget)

    def compute_param_importance(X, y):
        poly = PolynomialFeatures(degree=1, include_bias=False)
        scaler = StandardScaler()
        lasso = LassoCV(cv=3, random_state=seed, alphas=np.logspace(-4, 0, 100), max_iter=5000)
        model = make_pipeline(poly, scaler, lasso)
        model.fit(X, y)
        
        feature_names = poly.get_feature_names_out([f'p{i}' for i in range(D)])
        weights = lasso.coef_
        param_importance = np.zeros(D)
        for i in range(D):
            param_mask = [f'p{i}' in name for name in feature_names]
            param_importance[i] = np.sum(np.abs(weights[param_mask]))
        return param_importance

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    def get_current_knob_count(step):
        base = 4
        increment = 2
        interval = 5
        return min(base + (step // interval) * increment, D)

    while step < budget and consecutive_no_improve < maxlives:
        X_current = np.array([[independent_set[i].index(val) for i, val in enumerate(config)] for config in xs])
        y_current = np.array(results)
        X_combined = np.vstack([X_current, X_history]) if len(X_history) > 0 else X_current
        y_combined = np.hstack([y_current, y_history]) if len(y_history) > 0 else y_current

        param_importance = compute_param_importance(X_combined, y_combined)
        current_knob_count = get_current_knob_count(step)
        impactful_indices = np.argsort(param_importance)[-current_knob_count:]

        gp.fit(X_combined, y_combined)
        candidates = []
        while len(candidates) < 100:
            candidate = [random.choice(independent_set[i]) for i in range(D)]
            for idx in impactful_indices:
                if random.random() < 0.8:
                    candidate[idx] = random.choice(independent_set[idx])
            candidate_tuple = tuple(candidate)
            if candidate_tuple not in history_similar:
                candidates.append(candidate)

        X_candidates = np.array([[independent_set[i].index(val) for i, val in enumerate(conf)] for conf in candidates])
        mu, std = gp.predict(X_candidates, return_std=True)
        best_y = np.min(y_combined)
        ei = (best_y - mu) * norm.cdf((best_y - mu) / (std + 1e-9)) + std * norm.pdf((best_y - mu) / (std + 1e-9))
        best_idx = np.argmax(ei)
        next_original = candidates[best_idx]
        next_original_tuple = tuple(next_original)
        loop += 1

        next_score, similar_config = get_objective_score_with_similarity(dict_search, next_original)
        similar_tuple = tuple(similar_config)

        if similar_tuple in history_similar:
            consecutive_no_improve += 1
            continue

        history_original.add(next_original_tuple)
        history_similar.add(similar_tuple)
        xs.append(similar_config)
        results.append(next_score)
        step += 1

        if next_score < best_result:
            best_result = next_score
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    used_budget = min(step, budget)
    return (xs[:used_budget], results[:used_budget], 
            range(1, used_budget + 1), best_result, best_loop, used_budget)