from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict

pandas2ri.activate()

def scott_test(learning_models, compared_ress, smaller_is_better=True):
    data = pd.DataFrame({
        learning_models[i]: compared_ress[i]
        for i in range(len(learning_models))
    })

    try:
        sk = importr('ScottKnottESD')
        if smaller_is_better:
            data = data * -1
        r('set.seed(12345)')
        r_sk = sk.sk_esd(data, version="np")
        return r_sk
    except Exception as e:
        print(f"Error in Scott-Knott test: {e}")
        return None

def calculate_optimizer_stats_and_merge(problem_df, SEED_RANGE):
    optimizer_stats = problem_df.groupby('optimizer_name')['best_result'].agg(
        mean='mean',
        std=lambda x: x.std(ddof=1),
        min='min',
        max='max'
    ).reset_index()

    group_cols = ['mean', 'std', 'min', 'max']
    optimizer_stats['merged_group'] = optimizer_stats.groupby(group_cols)['optimizer_name'].transform(
        lambda x: '_'.join(sorted(x.tolist()))
    )

    merge_map = dict(zip(optimizer_stats['optimizer_name'], optimizer_stats['merged_group']))
    merged_stats = optimizer_stats.groupby('merged_group').first().reset_index()
    merged_stats_sorted = merged_stats.sort_values(
        by=['mean', 'std'], ascending=[True, True]
    ).reset_index(drop=True)

    pivot_df = problem_df.pivot(index='optimizer_name', columns='seed', values='best_result')
    for seed in SEED_RANGE:
        if seed not in pivot_df.columns:
            pivot_df[seed] = np.nan
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))

    merged_models = merged_stats_sorted['merged_group'].tolist()
    merged_ress = []
    for merged_name in merged_models:
        original_optimizer = optimizer_stats[optimizer_stats['merged_group'] == merged_name]['optimizer_name'].iloc[0]
        ress = pivot_df.loc[original_optimizer].values.tolist()
        merged_ress.append(ress)

    stats_df = merged_stats_sorted.rename(columns={'merged_group': 'optimizer_name'})[
        ['optimizer_name', 'mean', 'std', 'min', 'max']
    ]
    return stats_df, merged_models, merged_ress, merge_map

if __name__ == "__main__":
    csv_path = "result.csv"
    main_result_folder = "output/rq3"
    new_result_folder = "output/optimizer_rank_sub_files"
    SEED_RANGE = range(101, 131)
    MAX_RANK = 32

    ranking_csv = "output/rq3_ranking_results.csv"
    ranking_with_cat_csv = "output/rq3_ranking_with_categories.csv"
    optimizer_rank_csv = "output/rq3_optimizer_rank_by_domain_with_category.csv"
    category_summary_csv = "output/rq3_optimizer_category_rank_summary.csv"

    problem_category_csv = "problem_categories.csv"
    optimizer_category_csv = "optimizer_categories.csv"

    os.makedirs(main_result_folder, exist_ok=True)
    os.makedirs(new_result_folder, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        required_columns = ['optimizer_name', 'best_result', 'seed', 'problem_name']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
        df['best_result'] = pd.to_numeric(df['best_result'], errors='coerce')
        df = df.dropna(subset=['seed', 'best_result'])

        problem_names = df['problem_name'].unique().tolist()
        for problem in problem_names:
            problem_df = df[df['problem_name'] == problem].copy()
            problem_df = problem_df[problem_df['seed'].isin(SEED_RANGE)].copy()

            stats_df, merged_models, merged_ress, merge_map = calculate_optimizer_stats_and_merge(problem_df, SEED_RANGE)

            pd.set_option('display.float_format', lambda x: '%.6f' % x)

            if len(merged_models) < 2:
                result_content = "Insufficient optimizers for test"
            else:
                result = scott_test(merged_models, merged_ress, smaller_is_better=True)
                result_content = str(result) if result else "Test failed"

            fname = f"{problem}".replace("/", "_").replace("\\", "_").replace(":", "_") + ".txt"
            fpath = os.path.join(main_result_folder, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f"Problem: {problem}\n\n")
                f.write("Optimizer merge mapping:\n")
                for k, v in merge_map.items():
                    f.write(f"{k} → {v}\n")
                f.write("\nStatistics:\n")
                f.write(stats_df.to_string(index=True, float_format=lambda x: '%.6f' % x))
                f.write("\n\nScott-Knott result:\n")
                f.write(result_content)

    except Exception as e:
        print(f"Phase 1 error: {e}")

    all_results = []
    for filename in os.listdir(main_result_folder):
        if not filename.endswith(".txt"):
            continue
        problem = os.path.splitext(filename)[0]
        path = os.path.join(main_result_folder, filename)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]

            g_idx = None
            for i, line in enumerate(lines):
                if line.startswith("Groups:"):
                    g_idx = i
                    break
            if g_idx is None:
                continue

            content = lines[g_idx+1:]
            if len(content) % 2 != 0:
                continue

            optimizer_cluster = {}
            for i in range(0, len(content), 2):
                optimizers = re.split(r'\s+', content[i])
                clusters = re.split(r'\s+', content[i+1])
                for t, c in zip(optimizers, clusters):
                    if c.isdigit():
                        optimizer_cluster[t] = int(c)

            groups = {}
            for t, c in optimizer_cluster.items():
                groups.setdefault(c, []).append(t)
            sorted_groups = sorted(groups.items(), key=lambda x: x[0])
            ranked = ["_".join(sorted(ts)) for _, ts in sorted_groups]

            res = {"problem_name": problem}
            for i in range(MAX_RANK):
                res[f"rank_{i+1}"] = ranked[i] if i < len(ranked) else ""
            all_results.append(res)

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

    if all_results:
        pd.DataFrame(all_results).to_csv(ranking_csv, index=False)

    problem_cat = pd.read_csv(problem_category_csv)
    optimizer_cat = pd.read_csv(optimizer_category_csv)
    d_map = dict(zip(problem_cat['problem_name'], problem_cat['category']))
    t_map = dict(zip(optimizer_cat['optimizer_name'], optimizer_cat['category']))

    rank_df = pd.read_csv(ranking_csv)

    def add_dom(name):
        cat = d_map.get(name, 'unknown')
        return f"{name}({cat})"

    def add_tcat(val):
        if pd.isna(val) or val.strip() == "":
            return ""
        ts = val.split('_')
        return '_'.join([f"{t}({t_map.get(t, 'unknown')})" for t in ts])

    rank_df['problem_name'] = rank_df['problem_name'].apply(add_dom)
    rank_cols = [c for c in rank_df.columns if c.startswith('rank_')]
    for c in rank_cols:
        rank_df[c] = rank_df[c].apply(add_tcat)

    def get_key(name):
        match = re.search(r'\((.*?)\)', name)
        return match.group(1) if match else 'unknown'

    rank_df['_k'] = rank_df['problem_name'].apply(get_key)
    rank_df = rank_df.sort_values('_k').drop(columns=['_k'])
    rank_df.to_csv(ranking_with_cat_csv, index=False)

    df = pd.read_csv(ranking_with_cat_csv)
    t_cat_map = dict(zip(optimizer_cat['optimizer_name'], optimizer_cat['category']))
    rank_num = {c: int(c.split('_')[1]) for c in rank_cols}
    data = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        dm = re.search(r'\((HPO|SCT)\)', row['problem_name'])
        if not dm:
            continue
        dom = dm.group(1)
        for c in rank_cols:
            rk = rank_num[c]
            ts = row[c]
            if pd.isna(ts) or ts.strip() == "":
                continue
            for t in ts.split('_'):
                pure = re.sub(r'\(.*?\)', '', t)
                data[pure][dom].append(rk)
                data[pure]['all'].append(rk)

    res = []
    for optimizer, doms in data.items():
        cat = t_cat_map.get(optimizer, 'unknown')
        t_name = f"{optimizer}({cat})"

        a = doms.get('HPO', [])
        a_avg = round(np.mean(a), 2) if a else None
        a_std = round(np.std(a, ddof=1), 2) if a else None
        a_cnt = len(a)

        s = doms.get('SCT', [])
        s_avg = round(np.mean(s), 2) if s else None
        s_std = round(np.std(s, ddof=1), 2) if s else None
        s_cnt = len(s)

        all_l = doms.get('all', [])
        all_avg = round(np.mean(all_l), 2) if all_l else None
        all_std = round(np.std(all_l, ddof=1), 2) if all_l else None
        all_cnt = len(all_l)

        res.append({
            'optimizer_name_with_category': t_name,
            'hpo_domain_avg_rank': a_avg, 'hpo_domain_rank_std': a_std, 'hpo_domain_problem_count': a_cnt,
            'sct_domain_avg_rank': s_avg, 'sct_domain_rank_std': s_std, 'sct_domain_problem_count': s_cnt,
            'avg_rank': all_avg, 'rank_std': all_std, 'problem_count': all_cnt
        })

    res_df = pd.DataFrame(res).sort_values('hpo_domain_avg_rank')
    res_df.to_csv(optimizer_rank_csv, index=False)

    files = [
        (['optimizer_name_with_category', 'hpo_domain_avg_rank', 'hpo_domain_rank_std', 'hpo_domain_problem_count'], 'hpo_domain_avg_rank', 'optimizer_rank_hpo_domain.csv'),
        (['optimizer_name_with_category', 'sct_domain_avg_rank', 'sct_domain_rank_std', 'sct_domain_problem_count'], 'sct_domain_avg_rank', 'optimizer_rank_sct_domain.csv'),
        (['optimizer_name_with_category', 'avg_rank', 'rank_std', 'problem_count'], 'avg_rank', 'optimizer_rank_all_domains.csv')
    ]
    for cols, sort_col, fname in files:
        sub = res_df[cols].sort_values(sort_col)
        sub.to_csv(os.path.join(new_result_folder, fname), index=False)

    final_df = pd.read_csv(optimizer_rank_csv)
    def get_tcat(name):
        m = re.search(r'\((HPO|SCT|general)\)', name)
        return m.group(1) if m else 'unknown'
    final_df['optimizer_category'] = final_df['optimizer_name_with_category'].apply(get_tcat)

    summary = []
    for domain_col, domain_name in [('hpo_domain_avg_rank', 'hpo_domain'), ('sct_domain_avg_rank', 'sct_domain')]:
        sub = final_df[final_df[domain_col].notna()]
        for cat in ['HPO', 'SCT', 'general']:
            c_sub = sub[sub['optimizer_category'] == cat]
            avg = round(c_sub[domain_col].mean(), 2) if not c_sub.empty else None
            cnt = len(c_sub)
            summary.append({'domain': domain_name, 'optimizer_category': cat, 'avg_rank': avg, 'optimizer_count': cnt})

    pd.DataFrame(summary).to_csv(category_summary_csv, index=False)