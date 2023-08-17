import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygad

PROBLEM_CNT = 5000
MINER_CNT = 20
ITERATIONS = 2000
MIN_THRESHOLD = 0
MAX_THRESHOLD = 1
PACKET_SIZE = 10
MAX_ITERATIONS_BEFORE_REGEN = 1000
MIN_WIDTH = 0.4

def get_compute_powers(n):
    # Uniform distribution
    # return np.random.uniform(1, 5, n)

    # TEMP
    compute_pwr = np.full(n, 20)
    compute_pwr[0] = 40
    return compute_pwr

def get_difficulties_pareto(n):
    return np.random.pareto(2, n)*10

def regenerate_thresholds(thresholds, offsets, miner):
    # low, high = sorted(np.random.rand(2) * MAX_THRESHOLD)
    # # ensure that the difference between low and high is at least MIN_WIDTH
    # if high - low < MIN_WIDTH:
    #     high += MIN_WIDTH

    offsets[miner] = 0.4
    thresholds[miner] = 0.0
    # low[miner] = 0
    # high[miner] = 5

def get_tresholds():
    thresholds = np.zeros(MINER_CNT)
    offsets = np.zeros(MINER_CNT)
    for i in range(MINER_CNT):
        regenerate_thresholds(thresholds, offsets, i)
    return thresholds, offsets

def simulate_mining(thresholds, thresholds_offset):
    difficulties = get_difficulties_pareto(PROBLEM_CNT)
    compute_powers = get_compute_powers(MINER_CNT)
    rewards = np.zeros(MINER_CNT)

    # For each miner calculate how much time is needed to mine each problem and store in matrix
    remaining_times = np.array([difficulties / compute_powers[i] for i in range(MINER_CNT)])

    log_miner_total_times = []
    log_miner_packet_search_times = []

    for t in tqdm(range(ITERATIONS), desc='Iterations'):
        # print(f'Starting Iteration {t}...')
        miner_total_times = []
        miner_packet_search_times = []
        miner_packet = []
        for miner in range(MINER_CNT):
            # Find packet for each miner
            found_packet = False
            end_time = 0
            while not found_packet:
                end_time += 1 / compute_powers[miner]
                # Generate packets while the adequate one is not found
                packet = np.random.choice(range(PROBLEM_CNT), PACKET_SIZE, replace=False)
                
                high = thresholds[miner] + thresholds_offset[miner]
                if thresholds[miner] < remaining_times[miner][packet].sum() < high:
                    miner_packet_search_times.append(end_time)
                    miner_packet.append(packet)
                    end_time += remaining_times[miner][packet].sum()
                    miner_total_times.append(end_time)
                    found_packet = True
        
        miner_total_times = np.array(miner_total_times)
        miner_packet_search_times = np.array(miner_packet_search_times)

        # Find winner, update reward
        winner = np.argmin(miner_total_times)

        winner_time = miner_total_times[winner]
        winner_packet = miner_packet[winner]
        rewards[winner] += remaining_times[winner][winner_packet].sum()   # Assume that reward is linear function of difficulty

        # Reset difficulties for mined problems
        for instance in winner_packet:
            difficulties[instance] = get_difficulties_pareto(1)[0]

        # Update remaining times for all miners
        for miner in range(MINER_CNT):
            packet = miner_packet[miner]
            if miner_packet_search_times[miner] >= winner_time:
                continue

            # Update remaining times
            remaining_time_to_mine = winner_time - miner_packet_search_times[miner]
            for instance in packet:
                dt = min(remaining_time_to_mine, remaining_times[miner, instance])
                remaining_times[miner, instance] -= dt
                remaining_time_to_mine -= dt
                if remaining_time_to_mine <= 0:
                    break
            
            # Reset times for the block that was mined
            mined_packet = miner_packet[winner]
            for instance in mined_packet:
                remaining_times[miner, instance] = difficulties[instance] / compute_powers[miner]

        log_miner_total_times.append(miner_total_times)
        log_miner_packet_search_times.append(miner_packet_search_times)

    avg_block_times = np.array([np.mean([log_miner_total_times[i][j] for i in range(ITERATIONS)]) for j in range(MINER_CNT)])
    avg_search_times = np.array([np.mean([log_miner_packet_search_times[i][j] for i in range(ITERATIONS)]) for j in range(MINER_CNT)])

    return avg_block_times, avg_search_times, rewards

def plot():
    avg_block_times, avg_search_times, rewards = simulate_mining()
    # Sort by rewards
    sorted_ind = np.argsort(rewards)
    avg_block_times = avg_block_times[sorted_ind]
    avg_search_times = avg_search_times[sorted_ind]

    x = range(0,MINER_CNT*2,2)
    plt.bar(x, avg_search_times, width=1, color='r', label='Packet search times')
    plt.bar(x, avg_block_times, width=1, color='g', label='Block times', bottom=avg_search_times)
    plt.legend()
    plt.show()

def calc_fitness(ga_instance, solution, solution_idx):
    avg_block_times, avg_search_times, rewards = simulate_mining(thresholds=solution[:,0], thresholds_offset=solution[:,1])
    return rewards

def mutate(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))

        changed_gene = offspring[chromosome_idx, random_gene_idx]
        low = ga_instance.gene_space[random_gene_idx]['low']
        high = ga_instance.gene_space[random_gene_idx]['high']
        changed_gene += np.random.normal(0, 0.1)
        while not low <= changed_gene <= high:
            changed_gene = offspring[chromosome_idx, random_gene_idx]
            changed_gene += np.random.normal(0, 0.1)
        
        offspring[chromosome_idx, random_gene_idx] = changed_gene
            

    return offspring

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
tresholds = np.full(MINER_CNT, 0.0)
offsets = np.full(MINER_CNT, 3.5)
# Temp:
offsets[0] = np.inf
# tresholds[:MINER_CNT // 2] = 0.0
# offsets[:MINER_CNT // 2] = np.inf

avg_block_times, avg_search_times, rewards= simulate_mining(tresholds, offsets)
print()