import os
import csv
import math
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SPECIAL_PROBLEM_DIR = os.path.join(PROJECT_ROOT, "data", "HPO_problems", "multi-fidelity")
EXCLUDED_OPTIMIZERS = {}

def get_last_line_budget(csv_file, optimizer_name, is_special_problem):
    try:
        with open(csv_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                return None
        
        last_line = lines[-1]
        parts = last_line.split(',')
        if len(parts) < 1:
            return None
        
        target_optimizers = {'Hyperband', 'DEHB', 'BOHB'}
        if is_special_problem and optimizer_name in target_optimizers:
            if len(parts) >= 2:
                budget_str = parts[-2].strip()
            else:
                return None
        else:
            budget_str = parts[0].strip()
        
        try:
            return float(budget_str)
        except ValueError:
            return None
    except Exception:
        return None

def get_special_problems():
    if not os.path.exists(SPECIAL_PROBLEM_DIR):
        return set()
    
    special_problems = set()
    for filename in os.listdir(SPECIAL_PROBLEM_DIR):
        file_path = os.path.join(SPECIAL_PROBLEM_DIR, filename)
        if os.path.isfile(file_path):
            problem_name = os.path.splitext(filename)[0]
            special_problems.add(problem_name)
    
    return special_problems

def calculate_problem_budgets(result_dir, root_dir):
    special_problems = get_special_problems()
    
    seed_folders = [f for f in os.listdir(result_dir) 
                   if os.path.isdir(os.path.join(result_dir, f)) 
                   and f.isdigit() 
                   and 1 <= int(f) <= 5]
    
    if not seed_folders:
        return {}
    
    first_seed = seed_folders[0]
    first_seed_path = os.path.join(result_dir, first_seed)
    problem_names = [d for d in os.listdir(first_seed_path) 
                    if os.path.isdir(os.path.join(first_seed_path, d))]
    
    if not problem_names:
        return {}
    
    problematic_problems = []
    problem_optimizer_counts = defaultdict(dict)
    
    for problem in problem_names:
        issues = []
        
        for seed in seed_folders:
            problem_path = os.path.join(result_dir, seed, problem)
            if not os.path.exists(problem_path):
                issues.append(f"Seed {seed} missing problem folder")
                problem_optimizer_counts[problem][seed] = 0
                continue
            
            csv_files = []
            for f in os.listdir(problem_path):
                if os.path.isfile(os.path.join(problem_path, f)) and f.endswith('.csv'):
                    optimizer_name = f.split('_')[0]
                    if optimizer_name not in EXCLUDED_OPTIMIZERS:
                        csv_files.append(f)
            
            optimizer_count = len(csv_files)
            problem_optimizer_counts[problem][seed] = optimizer_count
            
            if optimizer_count == 0:
                issues.append(f"Seed {seed}: No valid CSV files")
        
        counts = list(problem_optimizer_counts[problem].values())
        if len(counts) > 0 and len(set(counts)) > 1:
            count_info = [f"Seed {s}: {c}" for s, c in problem_optimizer_counts[problem].items()]
            issues.append(f"Inconsistent optimizer counts: {', '.join(count_info)}")
        
        if issues:
            problematic_problems.append({
                "problem": problem,
                "issues": issues
            })
    
    problem_info = {}
    for problem in problem_names:
        is_special = problem in special_problems
        optimizer_budgets = defaultdict(list)
        
        for seed in seed_folders:
            problem_path = os.path.join(result_dir, seed, problem)
            if not os.path.exists(problem_path):
                continue
            
            csv_files = []
            for f in os.listdir(problem_path):
                if os.path.isfile(os.path.join(problem_path, f)) and f.endswith('.csv'):
                    optimizer_name = f.split('_')[0]
                    if optimizer_name not in EXCLUDED_OPTIMIZERS:
                        csv_files.append(f)
            
            for csv_file in csv_files:
                optimizer_name = csv_file.split('_')[0]
                csv_file_path = os.path.join(problem_path, csv_file)
                budget = get_last_line_budget(csv_file_path, optimizer_name, is_special)
                if budget is not None:
                    optimizer_budgets[optimizer_name].append((budget, seed))
        
        optimizer_max_info = {}
        for optimizer, budget_list in optimizer_budgets.items():
            if len(budget_list) == 5:
                max_budget, max_seed = max(budget_list, key=lambda x: x[0])
                optimizer_max_info[optimizer] = (max_budget, max_seed)
        
        if optimizer_max_info:
            max_optimizer = max(optimizer_max_info.items(), key=lambda x: x[1][0])
            best_optimizer, (final_budget, max_seed) = max_optimizer
            budget_ceil = math.ceil(final_budget)
            
            problem_info[problem] = {
                'final_budget': final_budget,
                'budget': budget_ceil,
                'source_optimizer': best_optimizer,
                'source_seed': max_seed
            }
    
    return problem_info

def save_budgets_to_csv(budgets_info, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'problem', 'final_budget', 'budget',
            'source_optimizer', 'source_seed'
        ])
        for problem, info in sorted(budgets_info.items()):
            writer.writerow([
                problem,
                info['final_budget'],
                info['budget'],
                info['source_optimizer'],
                info['source_seed']
            ])

if __name__ == "__main__":
    root_directory = PROJECT_ROOT
    result_directory = os.path.join(root_directory, 'result')
    
    if not os.path.exists(result_directory):
        exit(1)
    
    problem_info = calculate_problem_budgets(result_directory, root_directory)
    
    if problem_info:
        save_budgets_to_csv(problem_info, os.path.join(SCRIPT_DIR, 'budget.csv'))
