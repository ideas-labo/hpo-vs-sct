import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial
from optimizer.util.InferParamType import _infer_param_type
from typing import Dict, List, Tuple, Optional

def get_objective_score_with_similarity(dict_search: Dict[Tuple, float], config: List, param_types: Optional[List[str]] = None) -> Tuple[float, List]:
    config_tuple = tuple(config)
    if config_tuple in dict_search:
        return dict_search[config_tuple], list(config)

    if param_types is None:
        param_types = [_infer_param_type([config[i] for config in dict_search.keys()])
                      for i in range(len(next(iter(dict_search.keys()))))]

    continuous_indices = [i for i, t in enumerate(param_types) if t in ['continuous', 'discrete']]
    categorical_indices = [i for i, t in enumerate(param_types) if t == 'categorical']
    boolean_indices = [i for i, t in enumerate(param_types) if t == 'boolean']

    scaled_continuous = None
    scaler = None
    kdtree = None
    query_scaled = None
    config_list = list(dict_search.keys())

    full_scaled_continuous = None
    full_query_scaled = None

    if continuous_indices:
        continuous_configs = [
            [config[i] for i in continuous_indices]
            for config in config_list
        ]
        scaler = MinMaxScaler()
        scaled_continuous = scaler.fit_transform(continuous_configs)
        query_continuous = [config[i] for i in continuous_indices]
        query_scaled = scaler.transform([query_continuous])[0]

        full_scaled_continuous = scaled_continuous.copy()
        full_query_scaled = query_scaled.copy()

        kdtree = spatial.KDTree(scaled_continuous)

    def calculate_non_continuous_distance(key: Tuple) -> float:
        distance = 0
        if boolean_indices:
            bool_distance = sum(
                1 for i in boolean_indices if key[i] != config[i]
            )
            distance += bool_distance

        if categorical_indices:
            cat_distance = sum(
                1 for i in categorical_indices if key[i] != config[i]
            )
            distance += cat_distance
        return distance

    best_distance = float('inf')
    best_config = None
    best_value = None

    if kdtree is not None:
        try:
            k_neighbors = min(50, len(dict_search))
            distances, indices = kdtree.query(query_scaled, k=k_neighbors)

            if not isinstance(indices, np.ndarray):
                indices = np.array([indices])
                distances = np.array([distances])

            for idx, dist in zip(indices, distances):
                key = config_list[idx]
                other_distance = calculate_non_continuous_distance(key)
                total_distance = dist + other_distance

                if total_distance < best_distance:
                    best_distance = total_distance
                    best_config = key
                    best_value = dict_search[key]

            if best_config is not None:
                return best_value, list(best_config)
        except Exception as e:
            pass

    if continuous_indices:
        for idx, key in enumerate(config_list):
            continuous_distance = np.linalg.norm(full_scaled_continuous[idx] - full_query_scaled)
            other_distance = calculate_non_continuous_distance(key)
            total_distance = continuous_distance + other_distance

            if total_distance < best_distance:
                best_distance = total_distance
                best_config = key
                best_value = dict_search[key]
    else:
        for key in config_list:
            other_distance = calculate_non_continuous_distance(key)
            total_distance = other_distance

            if total_distance < best_distance:
                best_distance = total_distance
                best_config = key
                best_value = dict_search[key]

    return best_value, list(best_config)