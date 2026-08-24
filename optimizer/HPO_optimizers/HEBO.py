import warnings
import numpy as np
import pandas as pd
import random
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from optimizer.util.QueryProblem import get_objective_score_with_similarity
from typing import List
warnings.filterwarnings("ignore")

def run_optimizers(file, budget=100, seed=10, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)

    independent_set = file.independent_set
    dict_search = file.dict_search

    best_result = float('inf')
    best_loop = 0
    xs = []
    results = []
    history_raw_configs = set()
    history_similar_configs = dict()
    consecutive_no_improve = 0
    step = 0

    space_config = []
    for i, param_values in enumerate(independent_set):
        space_config.append({
                    'name': f'x{i}',
                    'type': 'cat',
                    'categories': param_values
                })
    space = DesignSpace().parse(space_config)

    opt = HEBO(space, rand_sample=0)

    initial_size = min(10, budget // 2)
    initial_X_list = []
    initial_y = []
    for i in range(initial_size):
        config_df = space.sample(1)
        config_array = config_df.iloc[0].values
        config_tuple = tuple(config_array)

        sample_count = 0
        while config_tuple in history_raw_configs and sample_count < 100:
            config_df = space.sample(1)
            config_array = config_df.iloc[0].values
            config_tuple = tuple(config_array)
            sample_count += 1
        if sample_count >= 100:
            continue

        current_score, similar_config = get_objective_score_with_similarity(dict_search, config_array)
        similar_config_tuple = tuple(similar_config)
        while similar_config_tuple in history_similar_configs and sample_count < 100:
            config_df = space.sample(1)
            config_array = config_df.iloc[0].values
            config_tuple = tuple(config_array)
            current_score, similar_config = get_objective_score_with_similarity(dict_search, config_array)
            similar_config_tuple = tuple(similar_config)
            sample_count += 1
        if sample_count >= 100:
            continue

        history_raw_configs.add(config_tuple)
        history_similar_configs[similar_config_tuple] = current_score
        initial_X_list.append(config_df)
        initial_y.append(current_score)
        xs.append(similar_config)
        results.append(current_score)
        step += 1

        if current_score < best_result:
            best_result = current_score
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    if initial_X_list:
        initial_X = pd.concat(initial_X_list, ignore_index=True)
        initial_y_np = np.array(initial_y).reshape(-1, 1)
        opt.observe(initial_X, initial_y_np)
    
    while step < budget and consecutive_no_improve < maxlives:
        rec = opt.suggest(n_suggestions=1)
        config_array = rec.iloc[0].values
        config_tuple = tuple(config_array)

        if config_tuple in history_raw_configs:
            consecutive_no_improve += 1
            continue

        current_score, similar_config = get_objective_score_with_similarity(dict_search, config_array)
        similar_config_tuple = tuple(similar_config)
        if similar_config_tuple in history_similar_configs:
            history_raw_configs.add(config_tuple)
            consecutive_no_improve += 1
            continue

        history_raw_configs.add(config_tuple)
        history_similar_configs[similar_config_tuple] = current_score
        y_np = np.array([[current_score]])
        
        opt.observe(rec, y_np)

        xs.append(similar_config)
        results.append(current_score)
        step += 1

        if current_score < best_result:
            best_result = current_score
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return (
        xs, 
        results, 
        range(1, step + 1),
        best_result, 
        best_loop, 
        step
    )