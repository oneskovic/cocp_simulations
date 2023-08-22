import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from packet_creation import PacketCreatorSimulated
import time
import numpy.typing as npt
from typing import Union, Callable
from multiprocessing import Pool


def get_compute_powers(n):
    return np.random.uniform(1, 5, n)


def get_difficulties_pareto(
    n: int,
    current_difficulties: Union[np.ndarray, None] = None,
    mined_problems: Union[np.ndarray, None] = None,
):
    return np.random.pareto(2, n)


def get_difficulties_uniform(
    n: int,
    current_difficulties: Union[np.ndarray, None] = None,
    mined_problems: Union[np.ndarray, None] = None,
):
    low = 0.01
    high = 3.0
    step = 0.01
    possible_difficulties = np.arange(low, high, step)
    if current_difficulties is None or mined_problems is None:
        return np.random.choice(possible_difficulties, n)
    else:
        # perfect_counts = np.full(len(possible_difficulties), len(current_difficulties) / len(possible_difficulties))
        # current_counts = np.histogram(current_difficulties, bins=np.append(possible_difficulties,high))[0]
        # diff = perfect_counts - current_counts
        # diff = diff.clip(0)
        # normalized_diff = diff / diff.sum()
        # return np.random.choice(possible_difficulties, n, p=normalized_diff)
        # Temp just return same difficulties
        return current_difficulties[mined_problems]


def get_tresholds(n):
    thresholds_low = np.full(n, 0.0)
    thresholds_high = np.full(n, 10.0)
    return thresholds_low, thresholds_high


class MiningSimulator:
    def __init__(
        self,
        problem_cnt: int,
        packet_size: int,
        miner_compute_powers: np.ndarray,
        miner_thresholds_low: np.ndarray,
        miner_thresholds_high: np.ndarray,
        iterations: int,
        packet_creator=PacketCreatorSimulated(),
        thread_cnt=8,
        difficulty_generator: Callable[
            [int, Union[np.ndarray, None], Union[np.ndarray, None]], np.ndarray
        ] = get_difficulties_uniform,
    ):
        self.difficulty_generator = difficulty_generator
        self.problem_difficulties = difficulty_generator(problem_cnt)
        self.packet_size = packet_size
        self.miner_compute_powers = miner_compute_powers
        self.miner_thresholds_low = miner_thresholds_low
        self.miner_thresholds_high = miner_thresholds_high
        self.iterations = iterations
        self.packet_creator = packet_creator
        self.thread_cnt = thread_cnt

        self.miner_cnt = len(miner_compute_powers)

    def get_fee(self, problem_difficulty):
        # Currently fee is proportional to difficulty
        return 5.0 + problem_difficulty

    def get_packet_search_times(self, packet_search_attempts: npt.NDArray[np.float64]):
        return packet_search_attempts / self.miner_compute_powers * 1e-5

    def get_packets(self, remaining_times):
        get_packet_args = [
            (
                remaining_times[miner],
                self.packet_size,
                self.miner_thresholds_low[miner],
                self.miner_thresholds_high[miner],
            )
            for miner in range(self.miner_cnt)
        ]

        # Split args into chunks
        chunk_size = self.miner_cnt // self.thread_cnt
        starmap_args = [
            get_packet_args[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self.thread_cnt)
        ]
        # Check if everything fits in chunks
        total_in_chunks = sum(len(chunk) for chunk in starmap_args)
        # If not add to last chunk
        if total_in_chunks < self.miner_cnt:
            starmap_args[-1] += get_packet_args[total_in_chunks:]      

        with Pool(self.thread_cnt) as p:
            packets_chunked = p.map(self.packet_creator.get_packet_chunked, starmap_args)
        packets = []
        for chunk in packets_chunked:
            packets += chunk

        # packets = [
        #     self.packet_creator.get_packet(
        #         remaining_times[miner],
        #         self.packet_size,
        #         self.miner_thresholds_low[miner],
        #         self.miner_thresholds_high[miner],
        #     )
        #     for miner in range(self.miner_cnt)
        # ]

        packet_problems, packet_search_times = zip(*packets)
        packet_problems = np.array(packet_problems)

        packet_search_times = np.array(packet_search_times, np.float64)
        packet_search_times = self.get_packet_search_times(packet_search_times)
        return packet_problems, packet_search_times

    def get_mine_times(self, remaining_times, packet_problems):
        return np.array(
            [
                remaining_times[miner, packet_problems[miner]].sum()
                for miner in range(self.miner_cnt)
            ]
        )

    def find_winner(self, miner_total_times, packet_problems):
        winner_time = miner_total_times.min()
        winners = np.argwhere(miner_total_times == winner_time).flatten()
        winner = np.random.choice(winners)
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
                self.packet_size, self.problem_difficulties, winner_packet
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
