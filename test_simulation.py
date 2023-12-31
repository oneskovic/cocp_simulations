import pytest
import numpy as np
from simulate_strategy import MiningSimulator
from packet_creation import PacketCreatorSimulated

@pytest.fixture
def mining_simulator():
    problem_cnt = 10
    packet_size = 2
    miner_cnt = 3
    miner_compute_powers = np.full(miner_cnt, 2)
    miner_thresholds_low = np.full(miner_cnt, 0.0)
    miner_thresholds_high = np.full(miner_cnt, 10.0)
    iterations = 1
    return MiningSimulator(problem_cnt, packet_size, miner_compute_powers, miner_thresholds_low, miner_thresholds_high, iterations)

@pytest.fixture
def packet_size():
    return 2

@pytest.fixture
def problem_difficulties():
    return np.array([4, 2, 5, 2, 6, 1, 0, 8, 9, 3])

@pytest.fixture
def remaining_times():
    return np.array(
        [
            [3, 5, 3, 1, 5, 8, 6, 3, 7, 2],
            [3, 8, 3, 6, 5, 6, 3, 9, 8, 6],
            [1, 3, 6, 4, 8, 0, 9, 4, 0, 7],
        ]
    )

@pytest.fixture
def packet_total_remaining_times(packet_problems, remaining_times):
    return remaining_times[
        np.arange(3)[:, None], packet_problems
    ].sum(axis=1)

@pytest.fixture
def packet_search_times():
    return np.array([1, 2, 3])

@pytest.fixture
def miner_total_times(packet_total_remaining_times, packet_search_times):
    return packet_total_remaining_times + packet_search_times

@pytest.fixture
def packet_problems():
    return np.array([[1, 2], [3, 4], [5, 6]])

@pytest.fixture
def packet_creator_simulated():
    return PacketCreatorSimulated()

def test_packet_creator_simulated(packet_creator_simulated, problem_difficulties, packet_size):
    threshold_low = 0.0
    threshold_high = 3.0
    packet, tries = packet_creator_simulated.get_packet(problem_difficulties, packet_size, threshold_low, threshold_high)
    assert tries > 0
    assert threshold_low <= problem_difficulties[packet].sum() <= threshold_high


def test_get_packets(mining_simulator, remaining_times):
    packet_problems, packet_search_times = mining_simulator.get_packets(remaining_times)
    assert packet_problems.shape == (mining_simulator.miner_cnt, mining_simulator.packet_size)
    assert packet_search_times.shape == (mining_simulator.miner_cnt,)
    assert np.all(packet_search_times > 0)    
    packet_total_remaining_times = remaining_times[
        np.arange(3)[:, None], packet_problems
    ].sum(axis=1)
    assert np.all(
        (mining_simulator.miner_thresholds_low <= packet_total_remaining_times)
        & (packet_total_remaining_times <= mining_simulator.miner_thresholds_high)
    )

def test_get_mine_times(mining_simulator, remaining_times, packet_problems):
    mine_times = mining_simulator.get_mine_times(remaining_times, packet_problems)
    assert np.all(mine_times == np.array([8, 11, 9]))

def test_find_winner(mining_simulator, miner_total_times, packet_problems):
    winner, winner_packet, winner_time, reward  = mining_simulator.find_winner(miner_total_times, packet_problems)
    assert winner == 0
    assert winner_time == 9
    assert np.all(winner_packet == packet_problems[winner])
    assert reward == sum(mining_simulator.get_fee(p) for p in mining_simulator.problem_difficulties[winner_packet])

def test_new_remaining_times(mining_simulator, packet_problems, packet_search_times, remaining_times):
    winner_time = 9
    new_remaining_times = mining_simulator.new_remaining_times(packet_problems, winner_time, packet_search_times, remaining_times)
    expected = np.array([
        [3, 0, 0, 1, 5, 8, 6, 3, 7, 2],
        [3, 8, 3, 0, 4, 6, 3, 9, 8, 6],
        [1, 3, 6, 4, 8, 0, 3, 4, 0, 7],
    ])
    assert np.all(new_remaining_times == expected)