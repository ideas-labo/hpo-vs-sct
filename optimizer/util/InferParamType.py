import numpy as np

def _infer_param_type(param_values):
    if not param_values:
        return 'categorical'

    has_string = any(isinstance(v, str) for v in param_values)
    if has_string:
        return 'categorical'

    all_numeric = all(isinstance(v, (int, float, np.number)) for v in param_values)
    if not all_numeric:
        return 'categorical'

    unique_values = list(set(param_values))
    num_unique = len(unique_values)
    num_total = len(param_values)
    unique_ratio = num_unique / num_total
    std_dev = np.std(param_values) if num_total > 1 else 0

    if unique_ratio > 0.1 and std_dev > 1:
        return 'continuous'
    elif num_unique <= 10:
        return 'discrete'
    else:
        return 'categorical'