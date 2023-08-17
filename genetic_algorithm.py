import pygad
from simulate_strategy import MiningSimulator
import numpy as np

# FIXME wtf
def calc_fitness(ga_instance, solution, solution_idx):
    ms = MiningSimulator()
    avg_block_times, avg_search_times, rewards = ms.run_simulation()
    return rewards

# FIXME wtf also
def mutate(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))

        changed_gene = offspring[chromosome_idx, random_gene_idx]
        low = ga_instance.gene_space[random_gene_idx]["low"]
        high = ga_instance.gene_space[random_gene_idx]["high"]
        changed_gene += np.random.normal(0, 0.1)
        while not low <= changed_gene <= high:
            changed_gene = offspring[chromosome_idx, random_gene_idx]
            changed_gene += np.random.normal(0, 0.1)

        offspring[chromosome_idx, random_gene_idx] = changed_gene

    return offspring

if __name__ == '__main__':
    # FIXME mrzi me
    # ga = pygad.GA(
    #     num_generations=200,
    #     num_parents_mating=int(MINER_CNT*0.3),
    #     fitness_func=calc_fitness,
    #     fitness_batch_size=MINER_CNT,
    #     initial_population=np.column_stack(get_tresholds()),
    #     random_seed=42,
    #     save_solutions=True,
    #     save_best_solutions=True,
    #     keep_elitism=0,
    #     keep_parents=0,
    #     gene_space=[{'low':0, 'high': MAX_THRESHOLD}, {'low':MIN_WIDTH, 'high': MAX_THRESHOLD}],
    #     mutation_type=mutate,
    # )
    # ga.run()
    # ga.plot_fitness()