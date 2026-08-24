import pandas as pd
import re
import os
from collections import defaultdict
import numpy as np

input_csv = "rank.csv"
domain_csv = "domain_problem_summary.csv"

output_folder = "output/optimizer_rank_by_domain"
os.makedirs(output_folder, exist_ok=True)

required_files = [input_csv, domain_csv]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} not found")

df = pd.read_csv(input_csv)
rank_columns = [col for col in df.columns if col.startswith('rank_')]
rank_number = {col: int(col.split('_')[1]) for col in rank_columns}

domain_df = pd.read_csv(domain_csv)
domain_to_problems = {}
for _, row in domain_df.iterrows():
    domain = row['domain'].strip()
    problems = [d.strip() for d in row['problems'].split(',')]
    domain_to_problems[domain] = problems

problem_to_domain = {}
for domain, problems in domain_to_problems.items():
    for d in problems:
        problem_to_domain[d] = domain

all_valid_problems = set(problem_to_domain.keys())

domain_optimizer_ranks = defaultdict(lambda: defaultdict(list))

for _, row in df.iterrows():
    problem_name = row['problem_name']
    pure_problem = re.sub(r'\s*\([as]\)\s*$', '', problem_name).strip()
    
    if pure_problem not in all_valid_problems:
        continue

    current_domain = problem_to_domain[pure_problem]

    for col in rank_columns:
        rank = rank_number[col]
        optimizers_str = row[col]
        
        if pd.isna(optimizers_str) or optimizers_str.strip() == "":
            continue
        
        optimizers = optimizers_str.split('_')
        for optimizer in optimizers:
            pure_optimizer = re.sub(r'\(.*?\)', '', optimizer)
            domain_optimizer_ranks[current_domain][pure_optimizer].append(rank)

for domain, optimizer_ranks in domain_optimizer_ranks.items():
    result = []
    for optimizer, ranks in optimizer_ranks.items():
        optimizer_name = optimizer
        
        avg_rank = round(np.mean(ranks), 2)
        std_rank = round(np.std(ranks, ddof=1), 2) if len(ranks) > 1 else 0.00
        count = len(ranks)
        
        result.append({
            'optimizer_name': optimizer_name,
            'avg_rank': avg_rank,
            'rank_std': std_rank,
            'problem_count': count
        })
    
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by='avg_rank', ascending=True)
    
    safe_domain = domain.replace(' ', '_').replace('/', '_')
    filename = f"optimizer_rank_{safe_domain}.csv"
    save_path = os.path.join(output_folder, filename)
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Generated: {filename}")

print("\nAll domain ranking files generated successfully")