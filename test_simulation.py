import pytest
import numpy as np
from simulate_strategy import MiningSimulator
from packet_creation import PacketCreatorSimulated

problem_cnt = 10
packet_size = 2
miner_cnt = 3
miner_compute_powers = np.full(miner_cnt, 2)
miner_thresholds_low = np.full(miner_cnt, 0.0)
miner_thresholds_high = np.full(miner_cnt, 100.0)
iterations = 1
simulator_args = [
    problem_cnt,
    packet_size,
    miner_compute_powers,
    miner_thresholds_low,
    miner_thresholds_high,
    iterations,
]


def test_packet_creator_simulated():
    problem_difficulties = np.array([4, 2, 5, 2, 6, 1, 0, 8, 9, 3])
    threshold_low = 0.0
    threshold_high = 3.0
    pc = PacketCreatorSimulated(problem_difficulties, packet_size)
    packet, tries = pc.get_packet(threshold_low, threshold_high)
    assert tries > 0
    assert threshold_low <= problem_difficulties[packet].sum() <= threshold_high


def test_get_packets():
    remaining_times = np.array(
        [
            [4, 2, 5, 2, 6, 1, 0, 8, 9, 3],
            [3, 8, 3, 6, 5, 6, 3, 9, 8, 6],
            [1, 3, 6, 4, 8, 0, 9, 4, 0, 7],
        ]
    )
    simulator = MiningSimulator(*simulator_args)
    packet_problems, packet_search_times = simulator.get_packets(remaining_times)
    assert packet_problems.shape == (miner_cnt, packet_size)
    assert packet_search_times.shape == (miner_cnt,)
    assert np.all(packet_search_times > 0.0)
    # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
    packet_difficulty = remaining_times[
        np.arange(miner_cnt)[:, None], packet_problems
    ].sum(axis=1)
    assert np.all(
        (miner_thresholds_low <= packet_difficulty)
        & (packet_difficulty <= miner_thresholds_high)
    )
