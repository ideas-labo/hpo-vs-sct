import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import random
import copy
import lingam
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from itertools import product
from causallearn.search.ConstraintBased.FCI import fci
from optimizer.util import QueryProblem
from optimizer.util.matrix import get_adjacency_matrix
from optimizer.util.helper import get_tree_paths, has_negative_value, in_rule, Model, merge_constraints
from optimizer.util.ei import get_ei
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass

def create_feature_mapping(features: List[str], independent_set: List[List[Any]]) -> dict:
    feature_mapping = {}
    for feat, values in zip(features, independent_set):
        has_string = any(isinstance(val, str) for val in values)
        if has_string:
            encode_map = {val: idx for idx, val in enumerate(values)}
            decode_map = {idx: val for val, idx in encode_map.items()}
            feature_mapping[feat] = {
                'encode': encode_map,
                'decode': decode_map
            }
        else:
            feature_mapping[feat] = None
    return feature_mapping

def encode_decision(decision: List[Any], features: List[str], feature_mapping: dict) -> List[Any]:
    encoded_decision = []
    for val, feat in zip(decision, features):
        mapping = feature_mapping[feat]
        if mapping is not None:
            try:
                encoded_val = mapping['encode'][val]
            except KeyError:
                encoded_val = 0
            encoded_decision.append(encoded_val)
        else:
            encoded_decision.append(val)
    return encoded_decision

def decode_decision(encoded_decision: List[Any], features: List[str], feature_mapping: dict) -> List[Any]:
    decoded_decision = []
    for val, feat in zip(encoded_decision, features):
        mapping = feature_mapping[feat]
        if mapping is not None:
            try:
                decoded_val = mapping['decode'][val]
            except KeyError:
                decoded_val = 'unknown'
            decoded_decision.append(decoded_val)
        else:
            decoded_decision.append(val)
    return decoded_decision

DEFAULT_ESTIMATORS = 10
DEFAULT_MAX_ITERATIONS = int(1e4)
DEFAULT_STOP_THRESHOLD = 0.01
DEFAULT_INITIAL_SIZE = 10
DEFAULT_MAX_LIVES = 200
DEFAULT_BUDGET = 100
DEFAULT_L_VALUE = 10
DEFAULT_K_VALUE = 0.1

@dataclass
class PromiseTuneConfig:
    initial_size: int = DEFAULT_INITIAL_SIZE
    maxlives: int = DEFAULT_MAX_LIVES
    budget: int = DEFAULT_BUDGET
    rule: bool = True
    l: int = DEFAULT_L_VALUE
    k: float = DEFAULT_K_VALUE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    stop_threshold: float = DEFAULT_STOP_THRESHOLD

def extract_unique_data(training_indep: List[List[Any]], training_dep: List[float]) -> Tuple[List[List[Any]], List[float]]:
    new_rows, new_perfs = [], []
    for row, perf in zip(training_indep, training_dep):
        if row not in new_rows: 
            new_rows.append(row)
            new_perfs.append(perf)
    return new_rows, new_perfs

def build_decision_tree_paths(training_indep: List[List[Any]], training_dep: List[float], feature_names: List[str], sample: int) -> List[Any]:
    model = RandomForestRegressor(n_estimators=DEFAULT_ESTIMATORS, min_samples_leaf=int(sample))
    model.fit(training_indep, training_dep)
    
    all_path = []
    for sub_tree in model.estimators_:
        paths = get_tree_paths(sub_tree, feature_names)
        for path in paths:
            if path not in all_path:
                all_path.append(path[:])
    return all_path

def cause_find(training_indep: List[List[Any]], training_dep: List[float], file: Any, sample: int, sample_plus: int = 0) -> Tuple[List[float], List[Any], int]:
    feature_names = file.features
    all_path = build_decision_tree_paths(training_indep, training_dep, feature_names, sample)
    new_rows, new_perfs = extract_unique_data(training_indep, training_dep)
    
    if len(all_path) >= len(new_rows) - 2:
        sample_plus += 1
        all_path = random.sample(all_path, len(new_rows) - 2)
    
    try:
        ACEs, valid_rules = model_fit(new_rows, all_path, feature_names, new_perfs)
    except Exception as e:
        if len(all_path) > 2:
            all_path = random.sample(all_path, len(all_path) - 2)
            try:
                ACEs, valid_rules = model_fit(new_rows, all_path, feature_names, new_perfs)
            except Exception as e2:
                return [], [], sample_plus
        else:
            return [], [], sample_plus
    
    return ACEs, valid_rules, sample_plus
    
def model_fit(new_rows: List[List[Any]], all_path: List[Any], feature_names: List[str], new_perfs: List[float]) -> Tuple[List[float], List[Any]]:
    column_names = copy.deepcopy(feature_names)
    training_indep_new = copy.deepcopy(new_rows)
    
    for i, x in enumerate(new_rows):
        for rule in all_path:
            training_indep_new[i].append(1 if in_rule(x, rule, feature_names) else 0)
        training_indep_new[i].append(new_perfs[i])
    
    labels = [f"rule{a}" for a in range(len(all_path))]
    column_names.extend(labels)
    labels.append("perf")
    column_names.append("perf")
    
    df = pd.DataFrame(training_indep_new, columns=column_names)
    df.drop(columns=feature_names, inplace=True)
    data = df.to_numpy()
    
    try:
        g, _ = fci(data, show_progress=False, verbose=False)
        adjacency_matrix = get_adjacency_matrix(g).T
        adjacency_matrix[:, -1] = 0
    except Exception as e:
        adjacency_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    
    prior_knowledge = adjacency_matrix
    prior_knowledge[-1, :-1] = 1
    prior_knowledge[-1, -1] = -1
    
    try:
        model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
        model.fit(data)
    except ValueError as e:
        return [], []
    
    valid_rules, ACEs = [], []
    for k in range(len(labels) - 1):
        ACE = model.estimate_total_effect(data, k, len(labels) - 1)
        if ACE:
            valid_rules.append(all_path[k])
            ACEs.append(ACE)

    return ACEs, valid_rules

def random_search_base(max_iterations: int, stop_threshold: float, estimators: List[Any], eta: float, x_generator, batch_size: int = 100):
    best_value = float('-inf')
    values, xs = [], []

    for batch_start in range(0, max_iterations, batch_size):
        batch_end = min(batch_start + batch_size, max_iterations)
        current_batch_size = batch_end - batch_start
        
        batch_configs = [x_generator() for _ in range(current_batch_size)]
        
        batch_predictions = []
        for estimator in estimators:
            batch_pred = estimator.predict(batch_configs)
            batch_predictions.append(batch_pred)
        
        batch_predictions = np.array(batch_predictions).T
        
        for i, current_x in enumerate(batch_configs):
            pred = batch_predictions[i]
            current_value = get_ei([pred], eta)[0]
            
            xs.append(current_x)
            values.append(current_value)
            
            if current_value > best_value:
                best_value = current_value
        
        if len(values) > batch_size and len(values) % (batch_size * 2) == 0:
            try:
                kde = stats.gaussian_kde(values)
                improvement_probability = 1 - kde.integrate_box_1d(-np.inf, best_value)
                if improvement_probability < stop_threshold:
                    break
            except Exception:
                break
    
    return xs, best_value

def random_search_with_static_distribution(max_iterations: int, stop_threshold: float, y: List[List[Any]], estimators: List[Any], eta: float, every_column: List[List[Any]]) -> Tuple[List[List[Any]], float]:
    def x_generator():
        current_x = list(random.choice(y))
        for index in range(len(current_x)):
            if current_x[index] == '_':
                current_x[index] = random.choice(every_column[index])
        return current_x
    
    return random_search_base(max_iterations, stop_threshold, estimators, eta, x_generator)

def random_search_with_static_distribution_cold(max_iterations: int, stop_threshold: float, y: List[List[Any]], estimators: List[Any], eta: float) -> Tuple[List[List[Any]], float]:
    def x_generator():
        return [random.choice(y[index]) for index in range(len(y))]
    
    return random_search_base(max_iterations, stop_threshold, estimators, eta, x_generator)

def get_training_sequence_by_PromiseTune(training_indep: List[List[Any]], training_dep: List[float], eta: float, file: Any, rule: bool, sample: int, k: float, sample_plus: int = 0) -> Tuple[List[List[Any]], Any, List[Any], int]:
    config = PromiseTuneConfig()
    model = Model(training_indep, training_dep)
    best_configs = sorted(zip(training_indep, training_dep), key=lambda x: x[1])[:int(k * len(training_indep))]
    estimators = model.estimators_
    every_column = file.independent_set
    header = file.features
    Rule, candidates = [], []
    
    if rule:
        ACEs, all_path, sample_plus = cause_find(training_indep, training_dep, file, sample + sample_plus, sample_plus)
        for i, path in zip(ACEs, all_path):
            if i < 0:
                Rule += path
                for indep, _ in best_configs:
                    if in_rule(indep, path, header):
                        break
    
    if (rule and not has_negative_value(ACEs)) or not rule:
        xs, _ = random_search_with_static_distribution_cold(config.max_iterations, config.stop_threshold, every_column, estimators, eta)
        candidates.extend(xs)
    else:
        for index in range(len(all_path)):
            if ACEs[index] > 0:
                continue
            
            results = [[] for _ in range(len(header))]
            single_rule = all_path[index]
            
            for i, column_values, single_feature in zip(range(len(header)), every_column, header):
                tmp_left, tmp_right = [], []
                for single_component in single_rule:
                    if single_component[0] == single_feature:
                        if single_component[2] == 'L':
                            tmp_left.append(single_component[1])
                        else:
                            tmp_right.append(single_component[1])
                
                if tmp_left and tmp_right: 
                    results[i] = [x for x in column_values if max(tmp_right) <= x <= min(tmp_left)]
                elif tmp_left:
                    results[i] = [x for x in column_values if x <= min(tmp_left)]
                elif tmp_right:
                    results[i] = [x for x in column_values if max(tmp_right) <= x]
            
            for i in range(len(results)):
                if not results[i]:
                    results[i] = ['_']
                elif len(results[i]) > 10:
                    results[i] = random.sample(results[i], 10)
            
            new_list = list(product(*results))
            xs, _ = random_search_with_static_distribution(config.max_iterations, config.stop_threshold, new_list, estimators, eta, every_column)
            candidates.extend(xs)
    
    return candidates, model, Rule, sample_plus

def get_best_configuration_id_PromiseTune(training_indep: List[List[Any]], training_dep: List[float], eta: float, file: Any, rule: bool, sample: int, k: float, sample_plus: int = 0) -> Tuple[Optional[List[Any]], List[Any], int]:
    test_sequence, model, Rule, sample_plus = get_training_sequence_by_PromiseTune(training_indep, training_dep, eta, file, rule, sample, k, sample_plus)
    estimators = model.estimators_
    
    pred = [e.predict(test_sequence) for e in estimators]
    value = get_ei(pred, eta)
    
    merged_predicted_objectives = [[i, a] for a, i in zip(value, test_sequence)]
    merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
    
    for x in merged_predicted_objectives:
        if x[0] not in training_indep:
            return x[0], Rule, sample_plus
    
    return None, Rule, sample_plus

def PromiseTune(file, config: PromiseTuneConfig, seed: int) -> Tuple[Optional[Any], List[Any]]:
    random.seed(seed)
    np.random.seed(seed)
    
    steps, sample_plus = 0, 0
    configuration_b = None
    results, x_axis, xs, Rules = [], [], [], []
    lives = config.maxlives
    best_loop = 0

    feature_mapping = create_feature_mapping(file.features, file.independent_set)
    
    encoded_independent_set = []
    for feat, values in zip(file.features, file.independent_set):
        mapping = feature_mapping[feat]
        if mapping is not None:
            encoded_vals = [mapping['encode'][v] for v in values]
            encoded_independent_set.append(encoded_vals)
        else:
            encoded_independent_set.append(values)
    file.independent_set = encoded_independent_set
    
    for sol in file.all_set + file.training_set + file.test_set:
        sol.decision = encode_decision(sol.decision, file.features, feature_mapping)
    
    encoded_dict_search = {}
    for dec_tuple, perf in file.dict_search.items():
        encoded_dec_tuple = tuple(encode_decision(dec_tuple, file.features, feature_mapping))
        encoded_dict_search[encoded_dec_tuple] = perf
    file.dict_search = encoded_dict_search

    training_indep = [t.decision for t in file.training_set]
    training_dep = []
    
    for action in training_indep:
        reward, configuration = QueryProblem.get_objective_score_with_similarity(file.dict_search, action)
        training_dep.append(reward)
        xs.append(action)
        results.append(reward)    
    
    result = min(results)
    exist_configuration = training_indep[:]
    
    while config.initial_size + steps < config.budget:
        steps += 1
        best_solution, Rule, sample_plus = get_best_configuration_id_PromiseTune(
            training_indep, training_dep, result, file, config.rule, config.l, config.k, sample_plus)
        
        if best_solution is None:
            break
            
        best_result, configuration = QueryProblem.get_objective_score_with_similarity(file.dict_search, best_solution)
        
        Rules.append(Rule)
        training_indep.append(best_solution)
        training_dep.append(best_result)
        
        if configuration in exist_configuration:
            config.budget += 1
            lives -= 1
        else:
            exist_configuration.append(configuration)
            x_axis.append(steps)
            xs.append(configuration)
            results.append(best_result)
            
            if best_result < result:
                configuration_b = configuration
                result = best_result
                lives = config.maxlives
                best_loop = len(exist_configuration)
            else:
                lives -= 1
        
        if lives == 0:
            break

    decoded_xs = []
    for encoded_dec in xs:
        encoded_dec_list = list(encoded_dec) if isinstance(encoded_dec, tuple) else encoded_dec
        decoded_dec = decode_decision(encoded_dec_list, file.features, feature_mapping)
        decoded_xs.append(decoded_dec)

    return decoded_xs, results, x_axis, result, best_loop , len(exist_configuration)

def run_optimizers(file, budget, seed, maxlives) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    config = PromiseTuneConfig()
    config.budget = budget
    config.maxlives = maxlives
    
    xs, results, x_axis, best_result, best_loop, used_budget = PromiseTune(
        file=file,
        config=config,
        seed=seed
    )
    
    return (
        np.array(xs, dtype=object),
        np.array(results),
        np.array(x_axis),
        best_result,
        best_loop,
        used_budget
    )