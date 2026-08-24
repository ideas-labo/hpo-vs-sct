import csv
import os
import inspect

def run_with_budget(optimizer_function, **kwargs):
    budget = kwargs.get('budget', 20)
    maxlives = kwargs.get('maxlives', 100)
    seed = kwargs.get('seed', 0)
    filename = kwargs.get('filename', '')
    file = kwargs.get('file', '')
    result = optimizer_function(file, budget, seed, maxlives)
    
    if isinstance(result, tuple):
        xs, results, x_axis, best_result, best_loop, used_budget = result

        func_file = inspect.getfile(optimizer_function)
        func_filename = os.path.splitext(os.path.basename(func_file))[0]
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        save_dir = os.path.join(project_root, 'result')
        
        seed_dir = os.path.join(save_dir, str(seed))
        problem_name = os.path.splitext(os.path.basename(filename))[0]
        problem_dir = os.path.join(seed_dir, problem_name)
        
        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)
        
        csv_file_path = os.path.join(problem_dir, f'{func_filename}_{problem_name}_seed{seed}.csv')

        with open(csv_file_path, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            for i, (config, score) in enumerate(zip(xs, results)):
                config_str = [str(c) for c in config]
                score_str = str(score)
                row = [str(i + 1)] + config_str + [score_str]
                csv_writer.writerow(row)

            is_integer = isinstance(best_loop, int) or (isinstance(best_loop, float) and abs(best_loop - round(best_loop)) < 1e-9)

            if is_integer:
                best_loop_int = round(best_loop)
                if best_loop_int - 1 < len(xs):
                    best_config = xs[best_loop_int - 1]
                    best_score = results[best_loop_int - 1]
                    best_loop_str = f"{best_loop:.0f}" if isinstance(best_loop, float) else str(best_loop_int)
                    best_config_str = [str(c) for c in best_config]
                    best_score_str = str(best_score)
                    last_row = [best_loop_str] + best_config_str + [best_score_str]
                    csv_writer.writerow(last_row)
            else:
                used = best_loop 
                matched_index = None
                
                for i, config in enumerate(xs):
                    if len(config) < 1:
                        continue
                    config_used_budget  = config[-1]
                    if config_used_budget  == used:
                        matched_index = i
                        break
                
                if matched_index is not None:
                    best_config = xs[matched_index]
                    best_score = results[matched_index]
                    best_config_str = [str(c) for c in best_config]
                    best_score_str = str(best_score)
                    last_row = [str(matched_index+1)] + best_config_str + [best_score_str]
                    csv_writer.writerow(last_row)

    return result
