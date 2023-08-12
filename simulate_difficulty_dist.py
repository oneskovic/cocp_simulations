import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Return >= 0
def get_difficulties_normal(n=100):
    return np.random.normal(0.5, 0.2, n).clip(0)
    
def get_difficulties_pareto(n=100):
    return np.random.pareto(2, n)

# Generate a sample where one node has large compute power others have equal
def get_compute_powers_one_large(large=0.05, others=0.01):
    n_ohters = (1-large) / others
    return np.array([large] + [others] * int(n_ohters))
    
ITERATIONS = 10000

compute_power = get_compute_powers_one_large(0.9,0.01)
cnt_wins = np.zeros_like(compute_power)
n = len(compute_power)
block_times = []

for _ in tqdm(range(ITERATIONS)):

    difficulties = get_difficulties_pareto(n)
    # plt.hist(difficulties)
    # plt.show()
    # Check who wins
    time_to_compute = difficulties / compute_power
    time_to_mine_block = np.min(time_to_compute)
    winner = np.argmin(time_to_compute)
    block_times.append(time_to_mine_block)
    cnt_wins[winner] += 1

plt.hist(block_times, bins=100)
plt.show()
print(f'Velika bela ima: {cnt_wins[0]/ITERATIONS*100:.2f}% pobeda, a ima {compute_power[0]*100:.2f}% računsku moć')
print(f'Maleni ima: {cnt_wins[1]/ITERATIONS*100:.2f}% pobeda, a ima {compute_power[1]*100:.2f}% računsku moć')
print()