import pandas as pd

NEW_VECTOR = [1,1,1,1,1,1,1,1]
NEW_VEC_STR = "1,1,1,1,1,1,1,1"

df = pd.read_csv("problem_vector_merged.csv")

def calculate_matches(problem_vec_str):
    d_vec = [int(i) for i in problem_vec_str.split(",")]
    count = 0
    for d, n in zip(d_vec, NEW_VECTOR):
        if d == n:
            count += 1
    return count

query_path = []
best_result = []

for match_num in range(8, 0, -1):
    matched_rows = []
    
    for _, row in df.iterrows():
        problem = row["problem"]
        vec_str = row["feature_vector"]
        optimizers = row["rank1_optimizers"]
        
        matched_count = calculate_matches(vec_str)
        
        if matched_count == match_num:
            matched_rows.append({
                "problem": problem,
                "vector": vec_str,
                "matched_count": matched_count,
                "rank1_optimizers": optimizers
            })
    
    query_path.append({
        "query_step": f"Match {match_num} features",
        "found_count": len(matched_rows),
        "details": matched_rows
    })
    
    if matched_rows:
        best_result = matched_rows
        break

print("New problem vector:", NEW_VEC_STR)
print("=" * 80)
print("Full query path (from 8 matches downward)")
print("=" * 80)

for step in query_path:
    status = "Found" if step["found_count"] > 0 else "No match"
    print(f"\n{step['query_step']}: {status} ({step['found_count']} entries)")
    
    for item in step["details"]:
        print(f"  Problem: {item['problem']}")
        print(f"  Vector: {item['vector']}")
        print(f"  Recommended optimizers: {item['rank1_optimizers']}\n")
    
    if step["found_count"] > 0:
        break

print("=" * 80)
print("Final recommendation (highest match)")
print("=" * 80)

if best_result:
    match_level = best_result[0]["matched_count"]
    all_optimizers = list(set([r["rank1_optimizers"] for r in best_result]))
    print(f"Highest match: {match_level} features")
    print(f"Recommended optimizers: {' | '.join(all_optimizers)}")
else:
    print("No match found")

log_entries = []
for step in query_path:
    if step["found_count"] > 0:
        for d in step["details"]:
            log_entries.append({
                "query_step": step["query_step"],
                "problem": d["problem"],
                "vector": d["vector"],
                "matched_count": d["matched_count"],
                "rank1_optimizers": d["rank1_optimizers"]
            })

if log_entries:
    log_df = pd.DataFrame(log_entries)
    log_df.to_csv("problem_query_path.csv", index=False, encoding="utf-8")
    print("\nQuery log saved to: problem_query_path.csv")