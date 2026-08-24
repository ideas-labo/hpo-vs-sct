import random
import numpy as np
from optimizer.util.QueryProblem import get_objective_score_with_similarity

def run_optimizers(file, budget=20, seed=0, maxlives=100):
    random.seed(seed)
    np.random.seed(seed)

    independent_set = file.independent_set
    dict_search = file.dict_search

    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []
    history_configs = {}
    loop = 0
    consecutive_no_improve = 0
    step = 0

    particles_position = []
    particles_similar_position = []
    particles_velocity = []
    particles_best_position = []
    particles_best_similar = []
    particles_best_score = []
    num_particles = 10

    particle_count = 0
    while particle_count < num_particles and step < budget and consecutive_no_improve < maxlives:
        loop += 1

        generated_position = [random.choice(values) for values in independent_set]
        
        current_score, similar_position = get_objective_score_with_similarity(dict_search, generated_position)
        similar_config_tuple = tuple(similar_position)

        if similar_config_tuple in history_configs:
            consecutive_no_improve += 1
            continue

        particle_count += 1
        step += 1

        history_configs[similar_config_tuple] = current_score

        particles_position.append(generated_position)
        particles_similar_position.append(similar_position)
        particles_velocity.append([0] * len(generated_position))
        particles_best_position.append(generated_position)
        particles_best_similar.append(similar_position)
        particles_best_score.append(current_score)

        if current_score < best_result:
            best_result = current_score
            best_config = similar_position
            best_loop = step
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

        xs.append(similar_position)
        results.append(current_score)

    max_iter = budget // num_particles if num_particles != 0 else 1
    if max_iter == 0:
        max_iter = 1

    iter_num = 0
    while iter_num < max_iter and step < budget and consecutive_no_improve < maxlives:
        iter_num += 1
        for i in range(num_particles):
            if step >= budget or consecutive_no_improve >= maxlives:
                break

            loop += 1

            new_generated_position = []
            for j in range(len(particles_position[i])):
                if random.random() < 0.3:
                    new_val = particles_best_position[i][j]
                elif random.random() < 0.5:
                    new_val = best_config[j]
                else:
                    new_val = random.choice(independent_set[j])
                new_generated_position.append(new_val)
            current_score, new_similar_position = get_objective_score_with_similarity(dict_search, new_generated_position)
            similar_config_tuple = tuple(new_similar_position)

            if similar_config_tuple in history_configs:
                consecutive_no_improve += 1
                continue

            step += 1

            history_configs[similar_config_tuple] = current_score

            particles_position[i] = new_generated_position
            particles_similar_position[i] = new_similar_position

            if current_score < particles_best_score[i]:
                particles_best_score[i] = current_score
                particles_best_position[i] = new_generated_position
                particles_best_similar[i] = new_similar_position

            if current_score < best_result:
                best_result = current_score
                best_config = new_similar_position
                best_loop = step
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            xs.append(new_similar_position)
            results.append(current_score)

    return xs, results, range(1, len(results) + 1), best_result, best_loop, step