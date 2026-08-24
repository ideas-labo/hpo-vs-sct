import pandas as pd
import numpy as np
import os
import re
import csv

# ===================== CONFIGURATION =====================
INPUT_FITNESS = "fitness_landscape.csv"
INPUT_TOP3 = "problem_top3.csv"

OUTPUT_INTERVALS = "final_metric_intervals.csv"
OUTPUT_LEVELS = "problem_level_vectors.csv"
OUTPUT_VECTOR_RANK1 = "problem_vector_with_rank1.csv"
OUTPUT_MERGED = "problem_vector_merged.csv"

METRIC_VALUE_RANGE = {
    "FDC": (-np.inf, np.inf),
    "FBD": (0, np.inf),
    "PLO": (0, 1),
    "NBC": (0, np.inf),
    "Skewness": (-np.inf, np.inf),
    "Kurtosis": (-np.inf, np.inf),
    "CL": (0, np.inf),
    "MIE": (0, np.inf)
}

# ===================== STEP 1: GENERATE METRIC INTERVALS =====================
def count_in_interval(series, interval_str):
    data = series.dropna()
    interval_str = interval_str.strip()
    left_sym = interval_str[0]
    right_sym = interval_str[-1]
    parts = interval_str[1:-1].split(',')
    lo_str = parts[0].strip()
    hi_str = parts[1].strip()

    lo = -np.inf if lo_str == '-∞' else float(lo_str)
    hi = np.inf if hi_str == '∞' else float(hi_str)

    if left_sym == '[' and right_sym == ']':
        mask = (data >= lo) & (data <= hi)
    elif left_sym == '[' and right_sym == ')':
        mask = (data >= lo) & (data < hi)
    elif left_sym == '(' and right_sym == ']':
        mask = (data > lo) & (data <= hi)
    else:
        mask = (data > lo) & (data < hi)
    return int(mask.sum())

def quantile_4_interval(metric_name, df):
    data = df[metric_name].dropna()
    total = len(data)
    if total == 0:
        return []

    q25 = data.quantile(0.25)
    q50 = data.quantile(0.50)
    q75 = data.quantile(0.75)
    g_min, g_max = METRIC_VALUE_RANGE[metric_name]

    intervals = [
        ("Low", "[", g_min, q25, "]"),
        ("Medium-Low", "(", q25, q50, "]"),
        ("Medium-High", "(", q50, q75, "]"),
        ("High", "(", q75, g_max, "]"),
    ]

    res = []
    for label, l_sym, L, R, r_sym in intervals:
        L_str = f"{L:.3f}" if np.isfinite(L) else "-∞"
        R_str = f"{R:.3f}" if np.isfinite(R) else "∞"
        interval_str = f"{l_sym}{L_str}, {R_str}{r_sym}"

        cnt = count_in_interval(data, interval_str)
        ratio = round(cnt / total * 100, 2) if total else 0

        res.append({
            "metric": metric_name,
            "label": label,
            "interval": interval_str,
            "sample_count": cnt,
            "sample_ratio(%)": ratio,
            "problem_count": cnt,
            "method": "Quartile 4-division",
            "source": "Q0/Q25/Q50/Q75"
        })
    return res

def generate_metric_intervals():
    if not os.path.exists(INPUT_FITNESS):
        raise FileNotFoundError(f"{INPUT_FITNESS} not found")
    df = pd.read_csv(INPUT_FITNESS)
    all_results = []

    for metric in METRIC_VALUE_RANGE.keys():
        if metric not in df.columns:
            print(f"Skipping {metric} (column not found)")
            continue
        res = quantile_4_interval(metric, df)
        all_results.extend(res)
        print(f"Processed: {metric}")

    cols = ["metric", "label", "interval", "sample_count", "sample_ratio(%)",
            "problem_count", "method", "source"]
    out_df = pd.DataFrame(all_results)[cols]
    out_df.to_csv(OUTPUT_INTERVALS, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_INTERVALS}")

# ===================== STEP 2: GENERATE LEVEL VECTORS =====================
def generate_level_vectors():
    df_intervals = pd.read_csv(OUTPUT_INTERVALS)
    df_fitness = pd.read_csv(INPUT_FITNESS)
    metric_rules = {}

    for metric, group in df_intervals.groupby("metric"):
        label_order = {"Low": 0, "Medium-Low": 1, "Medium-High": 2, "High": 3}
        group = group.sort_values("label", key=lambda x: x.map(label_order))
        intervals = []
        for _, row in group.iterrows():
            interval_str = row["interval"]
            match = re.match(r"([\[\(])(-?inf|\d+\.?\d*),\s*(-?inf|\d+\.?\d*)([\]\)])", interval_str.strip())
            if not match:
                continue
            lb, lv, rv, rb = match.groups()
            left = -float("inf") if lv == "-∞" else float(lv)
            right = float("inf") if rv == "∞" else float(rv)
            intervals.append({
                "left": left, "right": right,
                "left_closed": lb == "[", "right_closed": rb == "]"
            })
        metric_rules[metric] = intervals

    def value_to_level(v, m):
        if m not in metric_rules:
            return 4
        for i, itv in enumerate(metric_rules[m]):
            il = v >= itv["left"] if itv["left_closed"] else v > itv["left"]
            ir = v <= itv["right"] if itv["right_closed"] else v < itv["right"]
            if il and ir:
                return i + 1
        return 4

    metrics = ["FDC", "FBD", "PLO", "Skewness", "Kurtosis", "CL", "MIE", "NBC"]
    rows = []
    for _, row in df_fitness.iterrows():
        ds = row["problem"]
        lv = [value_to_level(row[m], m) for m in metrics]
        rows.append([ds] + lv)

    result_df = pd.DataFrame(rows, columns=["problem"] + [f"{m}_level" for m in metrics])
    result_df.to_csv(OUTPUT_LEVELS, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_LEVELS}")

# ===================== STEP 3: GENERATE VECTOR + RANK1 =====================
def generate_vector_with_rank1():
    df_intervals = pd.read_csv(OUTPUT_INTERVALS)
    df_fitness = pd.read_csv(INPUT_FITNESS)
    df_top3 = pd.read_csv(INPUT_TOP3)
    metric_rules = {}

    for metric, group in df_intervals.groupby("metric"):
        lo_map = {"Low": 0, "Medium-Low": 1, "Medium-High": 2, "High": 3}
        group = group.sort_values("label", key=lambda x: x.map(lo_map))
        intervals = []
        for _, row in group.iterrows():
            s = row["interval"]
            m = re.match(r"([\[\(])(-?inf|\d+\.?\d*),\s*(-?inf|\d+\.?\d*)([\]\)])", s.strip())
            if not m:
                continue
            lb, lv, rv, rb = m.groups()
            l = -float("inf") if lv == "-∞" else float(lv)
            r = float("inf") if rv == "∞" else float(rv)
            intervals.append({"left": l, "right": r, "lc": lb == "[", "rc": rb == "]"})
        metric_rules[metric] = intervals

    def v2l(v, m):
        if m not in metric_rules:
            return 4
        for i, itv in enumerate(metric_rules[m]):
            il = v >= itv["left"] if itv["lc"] else v > itv["left"]
            ir = v <= itv["right"] if itv["rc"] else v < itv["right"]
            if il and ir:
                return i + 1
        return 4

    metrics = ["FDC", "FBD", "PLO", "Skewness", "Kurtosis", "CL", "MIE", "NBC"]
    vdict = {}
    for _, row in df_fitness.iterrows():
        ds = row["problem"]
        vec = ",".join([str(v2l(row[m], m)) for m in metrics])
        vdict[ds] = vec

    df_result = df_top3.copy()
    df_result["feature_vector"] = df_result["problem"].map(vdict)
    out = df_result[["problem", "feature_vector", "rank1_optimizers"]]
    out.to_csv(OUTPUT_VECTOR_RANK1, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved: {OUTPUT_VECTOR_RANK1}")

# ===================== STEP 4: MERGE SAME VECTORS =====================
def merge_vectors():
    df = pd.read_csv(OUTPUT_VECTOR_RANK1)
    merged = df.groupby("feature_vector").agg({
        "problem": lambda x: "_".join(x),
        "rank1_optimizers": lambda x: "_".join(x)
    }).reset_index()
    out = merged[["problem", "feature_vector", "rank1_optimizers"]]
    out.to_csv(OUTPUT_MERGED, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_MERGED}")

# ===================== MAIN PIPELINE =====================
if __name__ == "__main__":
    print("=== START FULL PROCESSING ===")
    generate_metric_intervals()
    generate_level_vectors()
    generate_vector_with_rank1()
    merge_vectors()
    print("\n=== ALL FILES GENERATED SUCCESSFULLY ===")