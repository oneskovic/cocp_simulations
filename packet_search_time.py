import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import binom

PROBLEM_CNT = 100
PACKET_SIZE = 5
np.random.seed(42)

def get_packet(problems, packet_size):
    packet = np.random.choice(problems, packet_size, replace=False)
    return packet

def get_packet_search_time_simulated(low, high, problems, packet_size):
    found_packet = False
    tries = 0
    while not found_packet:
        tries += 1
        packet = get_packet(problems, packet_size)
        if low < packet.sum() < high:
            found_packet = True

    return tries

class PacketProbabilityCalculator:
    def __init__(self, problems, packet_size, rounding_interval=0.1):
        self.rounding_interval = rounding_interval
        rounded_problems = np.round(problems, decimals=int(-np.log10(rounding_interval)))
        self.dp = self.precompute(rounded_problems, packet_size)

    def precompute(self, problems, packet_size):
        # def solve_dp(elements, set_size, low, high, rounding_interval):
        # Map elements to ints in [0,1,...]
        problems = [int(round(x/self.rounding_interval)) for x in problems]

        n = len(problems)
        max_sum = sum(sorted(problems)[-packet_size:])
        dp = np.zeros((n+1,packet_size+1,max_sum+1), dtype=np.int64)
        for i in range(1,n+1):
            # Don't take element i (faster than to calculate in loop)
            dp[i] = dp[i-1].copy()
            # Take just element i
            dp[i,1,problems[i-1]] += 1
            for k in range(1,packet_size+1):
                for s in range(max_sum+1):
                    if s - problems[i-1] >= 0:
                        # Take element i -> number of sets with k-1 elements and sum s-elements[i-1]
                        dp[i,k,s] += dp[i-1,k-1,s-problems[i-1]]

        # sol = dp[-1][-1][low:high+1].sum()
        # return sol
        return dp[-1, -1, :] / binom(n, packet_size)
    
    def get_probability(self, low, high):
        low = int(round(low/self.rounding_interval))
        high = int(round(high/self.rounding_interval))
        return self.dp[low:high + 1].sum()
    
class PacketProbabilityCalculatorDumb:
    def __init__(self, problems, packet_size):
        self.problems = problems
        self.packet_size = packet_size
    
    def get_probability(self, low, high):
        ITERATIONS = 1000
        prob = 0
        for _ in range(ITERATIONS):
            packet = get_packet(self.problems, self.packet_size)
            if low < packet.sum() < high:
                prob += 1
        return prob / ITERATIONS

def get_packet_search_time_fast(low, high, problems, packet_size, ppc=PacketProbabilityCalculator):
    MAX_TRIES = 100
    packet_found_prob = ppc(problems, packet_size).get_probability(low, high)
    distribution = [np.power(1 - packet_found_prob, t - 1) * packet_found_prob for t in range(1, MAX_TRIES + 1)]
    distribution = np.array(distribution)
    # Sample from [1..n] according to distribution 
    tries = np.random.choice(np.arange(1, MAX_TRIES + 1), p=distribution)
    return tries


problems = np.random.pareto(2, PROBLEM_CNT)
low, high = 0, 5

ITERATIONS = 1000
times_simulated = []
times_calculated = []
    
for _ in tqdm(range(ITERATIONS)):
    t_sim = get_packet_search_time_simulated(low, high, problems, PACKET_SIZE)
    t_calc = get_packet_search_time_fast(low, high, problems, PACKET_SIZE)
    times_simulated.append(t_sim)
    times_calculated.append(t_calc)

times_simulated = np.array(times_simulated)
times_calculated= np.array(times_calculated)

max_tries = 20
def hist(a, bins):
    return [(a == i).sum() for i in range(1, bins + 1)]
print(hist(times_simulated, max_tries))
print(hist(times_calculated, max_tries))
