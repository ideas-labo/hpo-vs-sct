import random
import numpy as np
from optimizer.util.ReadProblem import get_data
from optimizer.util.QueryProblem import get_objective_score_with_similarity

def enum_to_numeric(enum_values):
    unique_enums = list(dict.fromkeys(enum_values))
    enum_map = {enum: i for i, enum in enumerate(unique_enums)}
    numeric_values = [enum_map[enum] for enum in enum_values]
    return numeric_values, enum_map, unique_enums

def numeric_to_enum(numeric_value, unique_enums):
    idx = int(round(numeric_value))
    idx = max(0, min(idx, len(unique_enums) - 1))
    return unique_enums[idx]

def divide_parameter_ranges(param_values, k):
    _, _, unique_enums = enum_to_numeric(param_values)
    return [[enum] for enum in unique_enums]

def generate_dds_samples(independent_set, num_samples):
    param_ranges = []
    param_enum_info = []

    for values in independent_set:
        numeric_values, enum_map, unique_enums = enum_to_numeric(values)
        param_enum_info.append((enum_map, unique_enums))
        ranges = divide_parameter_ranges(values, num_samples)
        param_ranges.append(ranges)

    samples = []
    for _ in range(num_samples):
        sample = []
        for p_idx, ranges in enumerate(param_ranges):
            range_idx = random.randint(0, len(ranges) - 1)
            current_range = ranges[range_idx]
            enum_value = current_range[0]
            sample.append(enum_value)
        samples.append(sample)
    return samples

def get_bounded_space(current_best, independent_set, other_samples, expand_factor=1):
    bounded_space = []
    for p_idx, (current_val, param_values) in enumerate(zip(current_best, independent_set)):
        _, enum_map, unique_enums = enum_to_numeric(param_values)
        current_numeric = enum_map[current_val]
        other_numerics = [enum_map[s[p_idx]] for s in other_samples if s[p_idx] in enum_map]
        sorted_others = sorted(other_numerics)
        global_min, global_max = 0, len(unique_enums) - 1

        left = global_min if not sorted_others else min(sorted_others)
        right = global_max if not sorted_others else max(sorted_others)

        left = max(global_min, left - expand_factor)
        right = min(global_max, right + expand_factor)

        candidates = [enum for enum in unique_enums if left <= enum_map[enum] <= right]
        candidates = candidates if candidates else unique_enums
        bounded_space.append(candidates)
    return bounded_space

def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)

    independent_set = file.independent_set
    dict_search = file.dict_search

    best_result = float('inf')
    best_config = None
    xs = []
    results = []
    history_configs = set()
    consecutive_no_improve = 0
    loop = 0
    best_loop = 0
    step = 0
    expand_factor = 1

    while step < budget and consecutive_no_improve < maxlives:
        if best_config is None:
            current_samples = generate_dds_samples(independent_set, num_samples=1)
        else:
            bounded_space = get_bounded_space(best_config, independent_set, xs, expand_factor)
            current_samples = generate_dds_samples(bounded_space, num_samples=1)
        current_config = current_samples[0]

        current_score, similar_config = get_objective_score_with_similarity(dict_search, current_config)
        similar_config_tuple = tuple(similar_config)

        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            loop += 1
            continue
        else:
            history_configs.add(similar_config_tuple)
            xs.append(similar_config)
            results.append(current_score)
            step += 1
            loop += 1

            if current_score < best_result:
                best_result = current_score
                best_config = similar_config
                best_loop = step
                consecutive_no_improve = 0
                expand_factor = 1
                improved = True
            else:
                consecutive_no_improve += 1
                improved = False

            if not improved:
                expand_factor = min(expand_factor * 2, 10)

    return xs, results, range(1, step + 1), best_result, best_loop, step