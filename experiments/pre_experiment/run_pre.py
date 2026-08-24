import os
import csv
import time
import traceback
from multiprocessing import Process, Queue, Manager
from optimizer.general_optimizers import RandomSearch, SA, HillClimbing, GA, BO, ES, CMAES, DE, PSO, ACO
from optimizer.SCT_optimizers import BestConfig, BOCA,ATConf, ROBOTune, ConEx, Tuneful, OtterTune, ResTune, FLASH, SWAY, PromiseTune
from optimizer.HPO_optimizers import HABO, HEBO, GGA, ParamILS, SMAC, irace, UQGuided, Hyperband, BOHB, DEHB
from optimizer.util.SaveToCSV import run_with_budget
from optimizer.util.ReadProblem import get_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULT_ROOT = os.path.join(PROJECT_ROOT, "result")
FIXED_BUDGET = 5000
PARALLEL_PROCESSES = 10
MAXLIVES = 100

OPTIMIZER_CATEGORY_MAP = {
    RandomSearch.run_optimizers: "general",
}

PROBLEM_FOLDERS = {
    os.path.join(PROJECT_ROOT, "data", "HPO_problems", "multi-fidelity"),
}

OPTIMIZERS = list(OPTIMIZER_CATEGORY_MAP.keys())

def get_seed_array_and_csv_name():
    seed_array = [1,2,3,4,5]
    csv_name = os.path.join(SCRIPT_DIR, "pre_result.csv")
    return seed_array, csv_name

SEED_ARRAY, csv_filename = get_seed_array_and_csv_name()

def get_optimizer_short_name(optimizer_func):
    return optimizer_func.__module__.split('.')[-1]

def result_exists(optimizer_category, problem_type, optimizer_name, problem_name, seed):
    problem_basename = os.path.splitext(problem_name)[0]
    old_result_path = os.path.join(
        RESULT_ROOT,
        str(seed),
        problem_basename,
        f"{optimizer_name}_{problem_basename}_seed{seed}.csv"
    )
    return os.path.exists(old_result_path)

def write_result(result):
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, mode='a', newline='') as csv_file:
        fieldnames = ["optimizer_category", "problem_type", "optimizer_name", "problem_name",
                      "best_result", "best_step", "used_budget", "error_message", "runtime", "seed"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def optimizer_worker(optimizer, budget, seed, maxlives, file, filename, queue):
    try:
        xs, scores, rounds, best_result, best_step, used_budget = run_with_budget(
            optimizer,
            budget=budget,
            seed=seed,
            maxlives=maxlives,
            file=file,
            filename=filename
        )
        queue.put(("success", best_result, best_step, used_budget))
    except Exception as e:
        error_msg = traceback.format_exc()
        queue.put(("error", str(e), error_msg))

def process_task(task, completed_tasks, running_tasks):
    problem_path, problem_type, optimizer, seed = task
    problem_name = os.path.basename(problem_path)
    optimizer_name = get_optimizer_short_name(optimizer)
    optimizer_category = OPTIMIZER_CATEGORY_MAP.get(optimizer, "unknown")
    task_id = (optimizer_category, problem_type, optimizer_name, problem_name, seed)

    if task_id in completed_tasks or task_id in running_tasks:
        return
    if result_exists(optimizer_category, problem_type, optimizer_name, problem_name, seed):
        completed_tasks.append(task_id)
        return

    current_budget = FIXED_BUDGET
    running_tasks.append(task_id)
    try:
        try:
            problem_data = get_data(problem_path, 10, 1)
        except Exception as e:
            error_msg = f"读取问题失败：{problem_path}，错误：{str(e)}"
            write_result({
                "optimizer_category": optimizer_category,
                "problem_type": problem_type,
                "optimizer_name": optimizer_name,
                "problem_name": problem_name,
                "best_result": None,
                "best_step": None,
                "used_budget": 0,
                "error_message": error_msg,
                "runtime": 0,
                "seed": seed
            })
            completed_tasks.append(task_id)
            return

        start_time = time.time()
        result_queue = Queue()
        process = Process(
            target=optimizer_worker,
            args=(optimizer, current_budget, seed, MAXLIVES, problem_data, problem_name, result_queue)
        )
        process.start()
        process.join()

        runtime = time.time() - start_time
        worker_result = result_queue.get()

        if worker_result[0] == "success":
            _, best_result, best_step, used_budget = worker_result
            write_result({
                "optimizer_category": optimizer_category,
                "problem_type": problem_type,
                "optimizer_name": optimizer_name,
                "problem_name": problem_name,
                "best_result": best_result,
                "best_step": best_step,
                "used_budget": used_budget,
                "error_message": None,
                "runtime": runtime,
                "seed": seed
            })
        elif worker_result[0] == "error":
            _, short_err, full_err = worker_result
            write_result({
                "optimizer_category": optimizer_category,
                "problem_type": problem_type,
                "optimizer_name": optimizer_name,
                "problem_name": problem_name,
                "best_result": None,
                "best_step": None,
                "used_budget": 0,
                "error_message": full_err,
                "runtime": runtime,
                "seed": seed
            })
        completed_tasks.append(task_id)

    except Exception as e:
        runtime = time.time() - start_time
        error_msg = traceback.format_exc()
        write_result({
            "optimizer_category": optimizer_category,
            "problem_type": problem_type,
            "optimizer_name": optimizer_name,
            "problem_name": problem_name,
            "best_result": None,
            "best_step": None,
            "used_budget": 0,
            "error_message": error_msg,
            "runtime": runtime,
            "seed": seed
        })
        completed_tasks.append(task_id)
    finally:
        if task_id in running_tasks:
            running_tasks.remove(task_id)

def generate_tasks_for_seed(seed):
    tasks = []
    for folder in PROBLEM_FOLDERS:
        if not os.path.exists(folder):
            continue
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        if not csv_files:
            continue
        problem_type = os.path.basename(folder)
        for csv_file in csv_files:
            problem_path = os.path.join(folder, csv_file)
            for optimizer in sorted(OPTIMIZERS, key=lambda x: get_optimizer_short_name(x)):
                tasks.append((problem_path, problem_type, optimizer, seed))
    return tasks

def worker_process(task_queue, completed_tasks, running_tasks):
    while not task_queue.empty():
        try:
            task = task_queue.get(block=False)
            process_task(task, completed_tasks, running_tasks)
        except Exception as e:
            time.sleep(1)

if __name__ == "__main__":
    total_start_time = time.time()
    for seed in SEED_ARRAY:
        tasks = generate_tasks_for_seed(seed)
        if not tasks:
            continue
        with Manager() as manager:
            task_queue = manager.Queue()
            for task in tasks:
                task_queue.put(task)
            completed_tasks = manager.list()
            running_tasks = manager.list()
            workers = []
            for i in range(PARALLEL_PROCESSES):
                p = Process(target=worker_process, args=(task_queue, completed_tasks, running_tasks))
                p.start()
                workers.append(p)
            for p in workers:
                p.join()
    total_runtime = time.time() - total_start_time
