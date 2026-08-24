import pandas as pd
import numpy as np
import math
import os
from pflacco.classical_ela_features import calculate_ela_distribution, calculate_information_content, calculate_nbc
from pflacco.misc_features import calculate_fitness_distance_correlation
from sklearn.preprocessing import LabelEncoder
import random


# --------------------------
# Parameter Type Inference
# --------------------------
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
    unique_ratio = num_unique / num_total if num_total > 0 else 0
    std_dev = np.std(param_values) if num_total > 1 else 0
    if unique_ratio > 0.1 and std_dev > 1:
        return 'continuous'
    elif num_unique <= 10:
        return 'discrete'
    else:
        return 'categorical'


# --------------------------
# Helper Functions
# --------------------------
def mixed_distance(config1, config2, param_types):
    hamming_sum = 0
    euclidean_sum = 0
    cat_count = 0
    cont_count = 0
    for val1, val2, p_type in zip(config1, config2, param_types):
        if p_type == 'categorical':
            if val1 != val2:
                hamming_sum += 1
            cat_count += 1
        else:
            euclidean_sum += (val1 - val2) ** 2
            cont_count += 1
    distance = 0
    if cat_count > 0:
        distance += hamming_sum / cat_count
    if cont_count > 0:
        distance += math.sqrt(euclidean_sum) / (cont_count ** 0.5)
    return distance


def find_value_indices(lst, is_minimize=True):
    if is_minimize:
        opt_value = min(lst)
    else:
        opt_value = max(lst)
    return [index for index, value in enumerate(lst) if value == opt_value]


def auto_correlation_at(x, lag=1):
    n = len(x)
    if n <= lag:
        return 0
    mean = np.mean(x)
    numerator = sum((x[i] - mean) * (x[i + lag] - mean) for i in range(n - lag))
    denominator = n - lag
    return numerator / denominator if denominator != 0 else 0


# --------------------------
# Fitness Landscape Analyzer
# --------------------------
class FitnessLandscapeAnalyzer:
    def __init__(self, data, effectiveness_col, is_minimize=True):
        self.data = data
        self.effectiveness_col = effectiveness_col
        self.series_perf = data[effectiveness_col]
        self.effectivenesss = self.series_perf.values
        self.is_minimize = is_minimize

        self.param_cols = [col for col in data.columns if col != effectiveness_col]
        self.df_params = data[self.param_cols]
        self.params = self.df_params.values
        self.n_samples, self.n_params = self.params.shape

        self.param_types = self._infer_all_param_types()
        print("Inferred parameter types:")
        for col, p_type in zip(self.param_cols, self.param_types):
            print(f"  {col}: {p_type}")

        self.opt_indices = find_value_indices(self.effectivenesss, self.is_minimize)
        self.opt_configs = [tuple(self.params[i]) for i in self.opt_indices]

    def _infer_all_param_types(self):
        return [
            _infer_param_type(self.data[col].tolist())
            for col in self.param_cols
        ]

    def calculate_FDC(self):
        fdc = calculate_fitness_distance_correlation(
            self.df_params,
            self.series_perf,
            minimize=self.is_minimize
        )
        return fdc['fitness_distance.fd_correlation']

    def calculate_FBD(self):
        if not self.opt_configs:
            return np.nan
        min_distances = []
        for config in self.params:
            dists = [mixed_distance(tuple(config), opt, self.param_types) for opt in self.opt_configs]
            min_distances.append(min(dists))
        return np.mean(min_distances)

    def calculate_PLO(self, max_neighbors=5):
        def is_better(x, y):
            return x <= y if self.is_minimize else x >= y

        def get_neighbors(config):
            neighbors = []
            for i in range(self.n_params):
                param_type = self.param_types[i]
                current_val = config[i]
                param_values = self.data[self.param_cols[i]].unique()

                if param_type == 'continuous':
                    sorted_vals = sorted(param_values)
                    idx = np.searchsorted(sorted_vals, current_val)
                    if idx >= len(sorted_vals) or sorted_vals[idx] != current_val:
                        idx = min(idx, len(sorted_vals) - 1)

                    candidates = []
                    start = max(0, idx - max_neighbors)
                    end = min(len(sorted_vals), idx + max_neighbors + 1)
                    for j in range(start, end):
                        if j != idx:
                            candidates.append(sorted_vals[j])
                else:
                    candidates = [v for v in param_values if v != current_val]
                    if len(candidates) > max_neighbors:
                        candidates = np.random.choice(candidates, max_neighbors, replace=False)

                for val in candidates:
                    neighbor = config.copy()
                    neighbor[i] = val
                    neighbors.append(tuple(neighbor))
            return neighbors

        local_opt_count = 0
        config_perf_map = {tuple(conf): perf for conf, perf in zip(self.params, self.effectivenesss)}

        for idx, config in enumerate(self.params):
            current_perf = self.effectivenesss[idx]
            is_local_opt = True
            for neighbor in get_neighbors(config):
                if neighbor in config_perf_map:
                    if not is_better(current_perf, config_perf_map[neighbor]):
                        is_local_opt = False
                        break
            if is_local_opt:
                local_opt_count += 1

        return local_opt_count / self.n_samples

    def calculate_skewness(self):
        ela_distr = calculate_ela_distribution(self.df_params, self.series_perf)
        return ela_distr['ela_distr.skewness']

    def calculate_kurtosis(self):
        ela_distr = calculate_ela_distribution(self.df_params, self.series_perf)
        return ela_distr['ela_distr.kurtosis']

    def calculate_CL(self):
        random.seed(42)
        np.random.seed(42)

        populations = [tuple(conf) for conf in self.params]
        base = self.opt_configs

        def _min_distance(x):
            dis = []
            for single_base in base:
                dist = mixed_distance(x, single_base, self.param_types)
                if dist != 0:
                    dis.append(dist)
            return min(dis) if dis else 1e8

        populations_sorted = sorted(populations, key=_min_distance)

        total_autocorr = 0.0
        sample_size = 50
        valid_samples = 0

        config_perf_map = {tuple(conf): perf for conf, perf in zip(self.params, self.effectivenesss)}

        for _ in range(sample_size):
            sub_list = base.copy()
            current = random.choice(base)

            for _ in range(50):
                k = 0.1
                neighbors = []
                while not neighbors or all(mixed_distance(conf, current, self.param_types) == 0 for conf in neighbors):
                    neighbors = [
                        conf for conf in populations_sorted
                        if mixed_distance(conf, current, self.param_types) <= k
                    ]
                    k += 0.1
                current = random.choice(neighbors)
                sub_list.append(current)

            fitness_values = [config_perf_map[conf] for conf in sub_list]
            n = len(fitness_values)
            if n < 2:
                continue

            autocorr = auto_correlation_at(fitness_values, lag=1)
            std = np.std(fitness_values)

            if std != 0:
                total_autocorr += autocorr / (std ** 2)
                valid_samples += 1

        if valid_samples == 0:
            return np.nan
        avg_autocorr = total_autocorr / valid_samples

        if avg_autocorr == 0 or abs(avg_autocorr) >= 1:
            return np.nan
        return -1.0 / math.log(abs(avg_autocorr))

    def calculate_MIE(self):
        ic = calculate_information_content(self.df_params, self.series_perf, seed=100)
        return ic['ic.h_max']

    def calculate_NBC(self):
        nbc = calculate_nbc(self.df_params, self.series_perf)
        return nbc['nbc.nn_nb.mean_ratio']


def encode_categorical_data(data):
    encoder = LabelEncoder()
    encoded_data = data.copy()

    for col in encoded_data.columns:
        if pd.api.types.is_bool_dtype(encoded_data[col]):
            encoded_data[col] = encoded_data[col].astype(int)
            print(f"Converted boolean column '{col}' to 0/1")
            continue

        if not pd.api.types.is_numeric_dtype(encoded_data[col]):
            encoded_data[col] = encoded_data[col].fillna('NaN')
            encoded_data[col] = encoder.fit_transform(encoded_data[col])
            print(f"Encoded column '{col}' to numeric values")

    return encoded_data


# --------------------------
# Main Function
# --------------------------
def main(folder_path='./problem'):
    os.makedirs('./results', exist_ok=True)
    summary_csv_path = os.path.join('./results', "fidelity0.csv")

    metric_columns = ["FDC", "FBD", "PLO", "Skewness", "Kurtosis", "CL", "MIE", "NBC"]
    all_columns = ["problem_name"] + metric_columns

    results_list = []
    if os.path.exists(summary_csv_path):
        try:
            existing_df = pd.read_csv(summary_csv_path)
            for col in all_columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            results_list = existing_df.to_dict('records')
            print(f"Loaded {len(results_list)} existing results")
        except Exception as e:
            raise RuntimeError(f"Failed to read existing results: {str(e)}") from e

    for filename in sorted(os.listdir(folder_path), key=str.lower):
        if filename.endswith('.csv'):
            problem_name = filename
            print(f"\n=== Processing file: {problem_name} ===")
            start_time = pd.Timestamp.now()

            existing_index = -1
            has_valid_result = False
            for i, item in enumerate(results_list):
                if item.get("problem_name") == problem_name:
                    existing_index = i
                    all_valid = True
                    for metric in metric_columns:
                        val = item.get(metric)
                        if pd.isna(val) or str(val).strip().lower() in ['nan', '']:
                            all_valid = False
                            break
                    has_valid_result = all_valid
                    break

            if has_valid_result:
                print("  Valid results already exist, skipping...")
                continue

            current_result = {col: None for col in all_columns}
            current_result["problem_name"] = problem_name
            if existing_index != -1:
                old_data = results_list[existing_index]
                for col in metric_columns:
                    val = old_data.get(col)
                    if not (pd.isna(val) or str(val).strip().lower() in ['nan', '']):
                        current_result[col] = val
                print("  Inherited valid metrics from previous record")

            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            data = encode_categorical_data(data)

            effectiveness_cols = [col for col in data.columns if col.startswith('$<')]
            if not effectiveness_cols:
                raise ValueError(f"No effectiveness column starting with '$<' found in {problem_name}")
            effectiveness_col = effectiveness_cols[0]
            is_minimize = True
            print(f"  Effectiveness column: {effectiveness_col}, Minimize: {is_minimize}")

            analyzer = FitnessLandscapeAnalyzer(
                data=data,
                effectiveness_col=effectiveness_col,
                is_minimize=is_minimize
            )

            metrics_func_map = {
                "FDC": analyzer.calculate_FDC,
                "FBD": analyzer.calculate_FBD,
                "PLO": analyzer.calculate_PLO,
                "Skewness": analyzer.calculate_skewness,
                "Kurtosis": analyzer.calculate_kurtosis,
                "CL": analyzer.calculate_CL,
                "MIE": analyzer.calculate_MIE,
                "NBC": analyzer.calculate_NBC
            }

            for metric, func in metrics_func_map.items():
                if current_result[metric] is None:
                    print(f"  Calculating {metric}...")
                    current_result[metric] = func()
                    val_str = round(current_result[metric], 4) if not pd.isna(current_result[metric]) else 'NaN'
                    print(f"  {metric}: {val_str}")

            elapsed = (pd.Timestamp.now() - start_time).round('SCT')
            print(f"  Processing completed in {elapsed}")

            if existing_index != -1:
                results_list[existing_index] = current_result
            else:
                results_list.append(current_result)

            summary_df = pd.DataFrame(results_list, columns=all_columns)
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"  Results saved to: {summary_csv_path}")

    print(f"\nAll files processed! Final output: {summary_csv_path}")


if __name__ == "__main__":
    main(folder_path='./sac')