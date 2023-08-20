import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from packet_creation import PacketCreatorSimulated
# Seed everything
np.random.seed(0)
random.seed(0)

# Return >= 0
def get_difficulties_normal(n=100):
    return np.random.normal(0.5, 0.2, n).clip(0)
    
def get_difficulties_pareto(n=100):
    return np.random.pareto(2, n)

def get_difficulties_uniform(n=100):
    return np.random.uniform(0.0, 3.0, n)

def get_difficulties_uniform_2(n=100):
    low = 0.01
    high = 3.0
    step = 0.01
    possible_difficulties = np.arange(low,high,step)
    return np.random.choice(possible_difficulties, n)

# Generate a sample where one node has large compute power others have equal
def get_compute_powers_one_large(large=0.05, others=0.01):
    n_ohters = (1-large) / others
    return np.array([large] + [others] * int(n_ohters))
    
ITERATIONS = 10000
PROBLEM_CNT = 1000

difficulties = get_difficulties_uniform_2(PROBLEM_CNT)
compute_power = get_compute_powers_one_large(0.05,0.01)
cnt_wins = np.zeros_like(compute_power)
n = len(compute_power)
block_times = []
miner_cnt = len(compute_power)

for _ in tqdm(range(ITERATIONS)):
    rand_diff = np.random.choice(difficulties, miner_cnt)
    miner_total_times =  rand_diff / compute_power
    packet_problems = [[np.argwhere(difficulties == diff)] for diff in rand_diff]

    time_to_compute = miner_total_times
    time_to_mine_block = np.min(time_to_compute)
    winner = np.argmin(time_to_compute)
    block_times.append(time_to_mine_block)
    cnt_wins[winner] += 1

# plt.hist(block_times, bins=100)
# plt.show()
print(f'Large has {cnt_wins[0]/ITERATIONS*100:.2f}% wins, and {compute_power[0]*100:.2f}% compute power')
print(f'Others have: {cnt_wins[1]/ITERATIONS*100:.2f}% wins, and {compute_power[1]*100:.2f}% compute power')
print()