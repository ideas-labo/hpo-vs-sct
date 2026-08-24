import random
import numpy as np
from optimizer.util.QueryProblem import get_objective_score_with_similarity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from pyDOE2 import lhs


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)
    
    independent_set = file.independent_set
    dict_search = file.dict_search
    n_params = len(independent_set)
    
    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_configs = set()
    consecutive_no_improve = 0
    loop = 0
    step = 0
    
    initial_samples = 10
    kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level_bounds=(1e-10, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if independent_set:
        fit_data = []
        for i, vals in enumerate(independent_set):
            for val in vals:
                row = [independent_set[j][0] for j in range(n_params)]
                row[i] = val
                fit_data.append(row)
        encoder.fit(fit_data)
        param_enc_ranges = [(0, len(vals)-1) for vals in independent_set]
    
    def normalize_encoded(encoded_data):
        norm_data = np.zeros_like(encoded_data, dtype=np.float64)
        for param_idx in range(n_params):
            min_enc, max_enc = param_enc_ranges[param_idx]
            if max_enc == min_enc:
                norm_data[:, param_idx] = 0.0
            else:
                norm_data[:, param_idx] = (encoded_data[:, param_idx] - min_enc) / (max_enc - min_enc)
        return norm_data
    
    def expected_improvement(X, X_sample, y_sample, gp, xi=0.01):
        X_encoded = encoder.transform(X)
        X_norm = normalize_encoded(X_encoded)
        mu, std = gp.predict(X_norm, return_std=True)
        
        mu_sample_opt = np.min(y_sample)
        imp = mu - mu_sample_opt - xi
        
        with np.errstate(divide='ignore'):
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0
        
        return ei
    
    def generate_lhs_initial_candidates(n_samples):
        lhs_continuous = lhs(n_params, samples=n_samples, criterion='maximin')
        lhs_encoded = np.zeros((n_samples, n_params), dtype=np.int64)
        for param_idx in range(n_params):
            min_enc, max_enc = param_enc_ranges[param_idx]
            lhs_encoded[:, param_idx] = np.round(lhs_continuous[:, param_idx] * (max_enc - min_enc) + min_enc)
            lhs_encoded[:, param_idx] = np.clip(lhs_encoded[:, param_idx], min_enc, max_enc)
        lhs_candidates = encoder.inverse_transform(lhs_encoded).tolist()
        return lhs_candidates
    
    while step < budget and consecutive_no_improve < maxlives:
        loop += 1
        
        if step < initial_samples:
            if step == 0:
                lhs_initial_candidates = generate_lhs_initial_candidates(initial_samples)
            current_candidate = lhs_initial_candidates[step]
        else:
            num_candidates = min(100, 50 + 10 * n_params)
            candidates = [
                [random.choice(values) for values in independent_set]
                for _ in range(num_candidates)
            ]
            
            unique_candidates = []
            seen_candidates = set()
            for cand in candidates:
                cand_tuple = tuple(cand)
                if cand_tuple not in seen_candidates:
                    seen_candidates.add(cand_tuple)
                    unique_candidates.append(cand)
            
            if len(unique_candidates) < num_candidates // 2:
                remaining = num_candidates - len(unique_candidates)
                unique_candidates.extend([
                    [random.choice(values) for values in independent_set]
                    for _ in range(remaining)
                ])
            
            candidates = unique_candidates
            
            X = np.array(xs)
            X_encoded = encoder.transform(X)
            X_norm = normalize_encoded(X_encoded)
            y = np.array(results)
            
            if len(X) > 20 and n_params > 10:
                _, X_sampled, _, y_sampled = train_test_split(
                    X_norm, y, test_size=0.7, random_state=seed
                )
                gp.fit(X_sampled, y_sampled)
            else:
                gp.fit(X_norm, y)
            
            ei_values = expected_improvement(candidates, X, y, gp)
            next_index = np.argmax(ei_values)
            current_candidate = candidates[next_index]
        
        current_score, similar_config = get_objective_score_with_similarity(dict_search, current_candidate)
        similar_config_tuple = tuple(similar_config)
        
        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            continue
        
        history_configs.add(similar_config_tuple)
        step += 1
        
        if current_score < best_result:
            best_result = current_score
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        xs.append(similar_config)
        results.append(current_score)
    
    return xs, results, range(1, len(results) + 1), best_result, best_loop, step