import numpy as np
from tqdm import tqdm
from scipy.special import binom


def get_rand_packet(problems, packet_size):
    packet = np.random.choice(np.arange(len(problems)), packet_size, replace=False)
    return packet


def is_valid_packet(problems, packet, low, high):
    return low <= problems[packet].sum() <= high


class PacketProbabilityCalculator:
    def __init__(self, problems, packet_size, rounding_interval=0.1):
        self.rounding_interval = rounding_interval
        rounded_problems = np.round(
            problems, decimals=int(-np.log10(rounding_interval))
        )
        self.dp = self.precompute(rounded_problems, packet_size)

    def precompute(self, problems, packet_size):
        # Map problems to ints in [0,1,...]
        problems = [int(round(x / self.rounding_interval)) for x in problems]

        n = len(problems)
        max_sum = sum(sorted(problems)[-packet_size:])
        dp = np.zeros((n + 1, packet_size + 1, max_sum + 1), dtype=np.int64)
        for i in range(1, n + 1):
            # Don't take problem i (faster than to calculate in loop)
            dp[i] = dp[i - 1].copy()
            # Take just problem i
            dp[i, 1, problems[i - 1]] += 1
            for k in range(1, packet_size + 1):
                for s in range(max_sum + 1):
                    if s - problems[i - 1] >= 0:
                        # Take problem i -> number of sets with k-1 problems and sum s-problems[i-1]
                        dp[i, k, s] += dp[i - 1, k - 1, s - problems[i - 1]]

        # sol = dp[-1][-1][low:high+1].sum()
        # return sol
        return dp[-1, -1, :] / binom(n, packet_size)

    def get_probability(self, low, high):
        low = int(round(low / self.rounding_interval))
        high = int(round(high / self.rounding_interval))
        return self.dp[low : high + 1].sum()


class PacketProbabilityCalculatorDumb:
    def __init__(self, problems, packet_size):
        self.problems = problems
        self.packet_size = packet_size

    def get_probability(self, low, high, iterations=1000):
        prob = 0
        for _ in range(iterations):
            packet = get_rand_packet(self.problems, self.packet_size)
            if is_valid_packet(self.problems, packet, low, high):
                prob += 1
        return prob / iterations


class PacketCreatorFast:
    def __init__(self, problems, packet_size, ppc=PacketProbabilityCalculator):
        self.ppc = ppc(problems, packet_size)

    def get_packet(self, low, high):
        MAX_TRIES = 100
        packet_found_prob = self.ppc.get_probability(low, high)
        distribution = [
            np.power(1 - packet_found_prob, t - 1) * packet_found_prob
            for t in range(1, MAX_TRIES + 1)
        ]
        distribution = np.array(distribution)
        # Sample from [1..n] according to distribution
        tries = np.random.choice(np.arange(1, MAX_TRIES + 1), p=distribution)
        return tries


class PacketCreatorSimulated:
    def __init__(self, problems, packet_size):
        self.problems = problems
        self.packet_size = packet_size

    def get_packet(self, low, high):
        found_packet = None
        tries = 0
        while found_packet is None:
            tries += 1
            packet = get_rand_packet(self.problems, self.packet_size)
            if is_valid_packet(self.problems, packet, low, high):
                found_packet = packet
        return found_packet, tries


if __name__ == "__main__":
    np.random.seed(42)

    PROBLEM_CNT = 100
    PACKET_SIZE = 5

    problems = np.random.pareto(2, PROBLEM_CNT)
    low, high = 0, 5

    ITERATIONS = 1000
    times_simulated = []
    times_calculated = []

    pst_fast = PacketCreatorFast(problems, PACKET_SIZE)
    pst_sim = PacketCreatorSimulated(problems, PACKET_SIZE)
    for _ in tqdm(range(ITERATIONS)):
        t_sim = pst_sim.get_packet(low, high)
        t_calc = pst_fast.get_packet_search_time(low, high)
        times_simulated.append(t_sim)
        times_calculated.append(t_calc)

    times_simulated = np.array(times_simulated)
    times_calculated = np.array(times_calculated)

    max_tries = 20
    hist = lambda a, bins: [(a == i).sum() for i in range(1, bins + 1)]
    print(hist(times_simulated, max_tries))
    print(hist(times_calculated, max_tries))
