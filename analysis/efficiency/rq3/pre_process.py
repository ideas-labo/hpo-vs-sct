from collections import defaultdict
import csv
import os

def get_complete_problems(problem_seed_optimizers, all_seeds, all_optimizers, expected_seeds=30, expected_optimizers=31):
    actual_seeds_count = len(all_seeds)
    actual_optimizers_count = len(all_optimizers)
    print(f"Complete Problem Criteria:")
    print(f"Expected: {expected_seeds} seeds × {expected_optimizers} optimizers")
    print(f"Actual: {actual_seeds_count} seeds × {actual_optimizers_count} optimizers")
    print("-" * 50)

    complete_problems = []
    incomplete_problems_detail = []

    for problem in sorted(problem_seed_optimizers.keys()):
        problem_related_seeds = problem_seed_optimizers[problem]
        missing_seeds = all_seeds - set(problem_related_seeds.keys())
        if missing_seeds:
            incomplete_problems_detail.append({
                "problem": problem,
                "reason": "Missing seeds",
                "details": f"Missing seeds: {sorted(list(missing_seeds))} (Total: {len(missing_seeds)})"
            })
            continue

        seed_missing_optimizers = {}
        for seed in all_seeds:
            optimizers_in_this_seed = problem_seed_optimizers[problem][seed]
            missing_optimizers = all_optimizers - optimizers_in_this_seed
            if missing_optimizers:
                seed_missing_optimizers[seed] = sorted(list(missing_optimizers))

        if seed_missing_optimizers:
            incomplete_problems_detail.append({
                "problem": problem,
                "reason": "Missing optimizers in seeds",
                "details": f"Missing details: {seed_missing_optimizers}"
            })
        else:
            complete_problems.append(problem)

    print(f"Found {len(complete_problems)} complete problems")
    print(f"Found {len(incomplete_problems_detail)} incomplete problems")
    return complete_problems, incomplete_problems_detail

def analyze_csv_for_complete_detection(csv_file_path):
    problem_seed_optimizers = defaultdict(lambda: defaultdict(set))
    all_seeds = set()
    all_optimizers = set()
    raw_data = []

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                print("Error: result.csv is empty")
                return None, None, None, None
            raw_data.append(header)

            for row_idx, row in enumerate(reader, 2):
                if not row:
                    continue
                if len(row) < 5:
                    print(f"Warning: Row {row_idx} invalid, skipped")
                    continue

                optimizer = row[2].strip()
                problem = row[3].strip()
                seed = row[-1].strip()

                if not all([optimizer, problem, seed]):
                    print(f"Warning: Row {row_idx} missing fields, skipped")
                    continue

                problem_seed_optimizers[problem][seed].add(optimizer)
                all_seeds.add(seed)
                all_optimizers.add(optimizer)
                raw_data.append(row)

        return problem_seed_optimizers, all_seeds, all_optimizers, raw_data

    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading {csv_file_path}: {str(e)}")
        return None, None, None, None

def load_category_maps():
    problem_cate_map = {}
    try:
        with open('problem_categories.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    problem_name = row[0].strip()
                    problem_type = row[1].strip()
                    problem_cate_map[problem_name] = problem_type
    except Exception as e:
        print(f"Failed to read problem_cate.csv: {e}")

    optimizer_cate_map = {}
    try:
        with open('optimizer_categories.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    optimizer_name = row[0].strip()
                    optimizer_category = row[1].strip()
                    optimizer_cate_map[optimizer_name] = optimizer_category
    except Exception as e:
        print(f"Failed to read optimizer_cate.csv: {e}")

    return optimizer_cate_map, problem_cate_map

def filter_complete_problems_from_result_csv(input_csv="result.csv", expected_seeds=30, expected_optimizers=31):
    problem_seed_optimizers, all_seeds, all_optimizers, raw_data = analyze_csv_for_complete_detection(input_csv)
    if None in [problem_seed_optimizers, all_seeds, all_optimizers, raw_data]:
        print("Data analysis failed, cannot filter")
        return

    complete_problems, _ = get_complete_problems(
        problem_seed_optimizers,
        all_seeds,
        all_optimizers,
        expected_seeds=expected_seeds,
        expected_optimizers=expected_optimizers
    )

    optimizer_cate_map, problem_cate_map = load_category_maps()

    filtered_raw_data = [raw_data[0]]
    complete_problems_set = set(complete_problems)

    for row in raw_data[1:]:
        if len(row) < 4:
            continue
        problem = row[3].strip()
        if problem not in complete_problems_set:
            continue

        optimizer_name = row[2].strip()
        optimizer_category = optimizer_cate_map.get(optimizer_name, "")
        problem_type = problem_cate_map.get(problem, "")

        row[0] = optimizer_category
        row[1] = problem_type
        filtered_raw_data.append(row)

    try:
        with open(input_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(filtered_raw_data)
        print(f"\nFilter completed!")
        print(f"Input: {input_csv} (Original rows: {len(raw_data)})")
        print(f"Output: {input_csv} (Filtered rows: {len(filtered_raw_data)})")
        print(f"Complete problems retained: {len(complete_problems)}")
    except Exception as e:
        print(f"Failed to save result: {str(e)}")

if __name__ == "__main__":
    filter_complete_problems_from_result_csv(
        input_csv="result.csv",
        expected_seeds=30,
        expected_optimizers=31
    )