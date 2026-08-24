import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from optimizer.util.QueryProblem import get_objective_score_with_similarity


def get_dynamic_d(D):
    if D <= 5:
        return D - 1
    elif 5 < D <= 20:
        return max(5, int(D * 0.6))
    else:
        return 20


def encode_config(config, encoders):
    encoded = []
    for idx, value in enumerate(config):
        if isinstance(value, (str, bool)):
            encoded.append(encoders[idx][value])
        else:
            encoded.append(encoders[idx][value])
    return encoded


def decode_config(encoded_config, decoders):
    decoded = []
    for idx, value in enumerate(encoded_config):
        decoded.append(decoders[idx][int(round(value))])
    return decoded


def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)
    p = 0.8
    independent_set = file.independent_set
    dict_search = file.dict_search
    D = len(independent_set)
    d = get_dynamic_d(D)

    encoders = []
    decoders = []
    for dim_values in independent_set:
        unique_vals = []
        for v in dim_values:
            if v not in unique_vals:
                unique_vals.append(v)
        encoder = {v: i for i, v in enumerate(unique_vals)}
        decoder = unique_vals
        encoders.append(encoder)
        decoders.append(decoder)

    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    encoded_xs = []
    results = []
    history_configs = dict()
    step = 0
    loop = 0
    consecutive_no_improve = 0

    initial_samples = min(5, budget)
    while step < initial_samples and consecutive_no_improve < maxlives:
        loop += 1
        current_config = [random.choice(values) for values in independent_set]

        current_score, similar_config = get_objective_score_with_similarity(dict_search, current_config)
        similar_config_tuple = tuple(similar_config)

        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        history_configs[similar_config_tuple] = current_score
        encoded_similar = encode_config(similar_config, encoders)
        xs.append(similar_config)
        encoded_xs.append(encoded_similar)
        results.append(current_score)
        step += 1

        if current_score < best_result:
            best_result = current_score
            best_config = similar_config
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    if step >= budget or consecutive_no_improve >= maxlives:
        used_budget = min(step, budget)
        return xs[:used_budget], results[:used_budget], range(1, used_budget + 1), best_result, best_loop, used_budget

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

    while step < budget and consecutive_no_improve < maxlives:
        loop += 1
        selected_indices = random.sample(range(D), d)
        dropped_indices = [i for i in range(D) if i not in selected_indices]

        X_selected = np.array([[config[i] for i in selected_indices] for config in encoded_xs])
        y = np.array(results)
        gp.fit(X_selected, y)

        num_candidates = 100
        candidates_selected = []
        for _ in range(num_candidates):
            raw_candidate = [random.choice(independent_set[i]) for i in selected_indices]
            encoded_candidate = encode_config(raw_candidate, [encoders[i] for i in selected_indices])
            candidates_selected.append(encoded_candidate)

        X_candidate = np.array(candidates_selected)
        mu, std = gp.predict(X_candidate, return_std=True)
        best_y = np.min(y)
        imp = best_y - mu
        Z = imp / (std + 1e-9)
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        best_candidate_idx = np.argmax(ei)
        best_encoded_selected = candidates_selected[best_candidate_idx]

        new_encoded_config = [None] * D
        for idx, sel_idx in enumerate(selected_indices):
            new_encoded_config[sel_idx] = best_encoded_selected[idx]

        current_best_encoded = encoded_xs[np.argmin(results)]
        for drop_idx in dropped_indices:
            if random.random() < p:
                new_encoded_config[drop_idx] = current_best_encoded[drop_idx]
            else:
                raw_rand = random.choice(independent_set[drop_idx])
                new_encoded_config[drop_idx] = encode_config([raw_rand], [encoders[drop_idx]])[0]

        new_config = decode_config(new_encoded_config, decoders)

        new_score, similar_config = get_objective_score_with_similarity(dict_search, new_config)
        similar_config_tuple = tuple(similar_config)

        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        history_configs[similar_config_tuple] = new_score
        encoded_similar = encode_config(similar_config, encoders)
        xs.append(similar_config)
        encoded_xs.append(encoded_similar)
        results.append(new_score)
        step += 1

        if new_score < best_result:
            best_result = new_score
            best_config = similar_config
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    used_budget = min(step, budget)
    return xs[:used_budget], results[:used_budget], range(1, used_budget + 1), best_result, best_loop, used_budget