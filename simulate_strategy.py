import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from packet_creation import PacketCreatorSimulated
import time

def get_compute_powers(n):
    return np.random.uniform(1, 5, n)


def get_difficulties_pareto(n):
    return np.random.pareto(2, n)


def get_tresholds(n):
    thresholds_low = np.full(n, 0.0)
    thresholds_high = np.full(n, 10.0)
    return thresholds_low, thresholds_high


class MiningSimulator:
    def __init__(
        self,
        problem_cnt,
        packet_size,
        miner_compute_powers,
        miner_thresholds_low,
        miner_thresholds_high,
        iterations,
        packet_creator=PacketCreatorSimulated,
        difficulty_generator=get_difficulties_pareto,
    ):
        self.difficulty_generator = difficulty_generator
        self.problem_difficulties = difficulty_generator(problem_cnt)
        self.packet_size = packet_size
        self.miner_compute_powers = miner_compute_powers
        self.miner_thresholds_low = miner_thresholds_low
        self.miner_thresholds_high = miner_thresholds_high
        self.iterations = iterations
        self.packet_creator = packet_creator

        self.miner_cnt = len(miner_compute_powers)

    def get_fee(self, problem_difficulty):
        # Currently fee is proportional to difficulty
        return problem_difficulty

    def get_packets(self, remaining_times):
        packets = [
            self.packet_creator(remaining_times[miner], self.packet_size).get_packet(
                self.miner_thresholds_low[miner], self.miner_thresholds_high[miner]
            )
            for miner in range(self.miner_cnt)
        ]
        packet_problems, packet_search_times = zip(*packets)
        packet_problems = np.array(packet_problems)

        packet_search_times = np.array(packet_search_times, np.float64)
        packet_search_times /= self.miner_compute_powers
        return packet_problems, packet_search_times

    def get_mine_times(self, remaining_times, packet_problems):
        return np.array(
            [
                remaining_times[miner, packet_problems[miner]].sum()
                for miner in range(self.miner_cnt)
            ]
        )

    def find_winner(self, miner_total_times, packet_problems):
        winner = np.argmin(miner_total_times)
        winner_time = miner_total_times[winner]
        winner_packet = packet_problems[winner]
        reward = self.get_fee(self.problem_difficulties[winner_packet]).sum()
        return winner, winner_packet, winner_time, reward

    def new_remaining_times(
        self, packet_problems, winner_time, packet_search_times, remaining_times
    ):
        new_remaining_times = remaining_times.copy()
        for miner in range(self.miner_cnt):
            packet = packet_problems[miner]
            remaining_time_to_mine = winner_time - packet_search_times[miner]
            for problem in packet:
                if remaining_time_to_mine <= 0:
                    break

                dt = min(remaining_time_to_mine, new_remaining_times[miner, problem])
                new_remaining_times[miner, problem] -= dt
                remaining_time_to_mine -= dt
        return new_remaining_times

    def run_simulation(self):
        rewards = np.zeros(self.miner_cnt)
        block_miner_packet_search_times = []
        block_miner_packet_mine_times = []

        # For each miner calculate how much time is needed to mine each problem and store in matrix
        remaining_times = [
            self.problem_difficulties / self.miner_compute_powers[miner]
            for miner in range(self.miner_cnt)
        ]
        remaining_times = np.array(remaining_times)

        for _ in tqdm(range(self.iterations), desc="Iterations"):
            packet_problems, packet_search_times = self.get_packets(remaining_times)
            packet_mine_times = self.get_mine_times(remaining_times, packet_problems)
            miner_total_times = packet_search_times + packet_mine_times

            winner, winner_packet, winner_time, winner_reward = self.find_winner(
                miner_total_times, packet_problems
            )
            rewards[winner] += winner_reward

            # Reset difficulties for mined problems
            self.problem_difficulties[winner_packet] = self.difficulty_generator(
                self.packet_size
            )

            # Update remaining times for all miners
            remaining_times = self.new_remaining_times(
                packet_problems, winner_time, packet_search_times, remaining_times
            )
            # Reset times for the block that was mined
            for miner in range(self.miner_cnt):
                remaining_times[miner, winner_packet] = (
                    self.problem_difficulties[winner_packet]
                    / self.miner_compute_powers[miner]
                )

            block_miner_packet_search_times.append(packet_search_times)
            block_miner_packet_mine_times.append(packet_mine_times)

        search_times = np.array(block_miner_packet_mine_times)
        mine_times = np.array(block_miner_packet_search_times)
        return search_times, mine_times, rewards


if __name__ == "__main__":
    PROBLEM_CNT = 200
    MINER_CNT = 20
    ITERATIONS = 1000
    PACKET_SIZE = 10
    difficulties = get_difficulties_pareto(PROBLEM_CNT)
    compute_powers = get_compute_powers(MINER_CNT)
    thresholds_low, thresholds_high = get_tresholds(MINER_CNT)
    search_times, mine_times, rewards = MiningSimulator(
        PROBLEM_CNT,
        PACKET_SIZE,
        compute_powers,
        thresholds_low,
        thresholds_high,
        ITERATIONS,
    ).run_simulation()
    plt.bar(np.arange(MINER_CNT), rewards)
    plt.show()
