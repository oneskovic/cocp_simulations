# Setup
Install the required packages:
```
pip install -r requirements.txt
```
# Description of simulator
The simulator is a simple environment that simulates strategies of miners on the network as proposed in the proof of useful work paper. 
## Assumptions
The simulator is based on the following assumptions:
1. Each miner follows this strategy:
    - Decide which package treshold is acceptable
    - Look for a package that fits the treshold
    - Start mining the package
    - Once someone finds a new block, stop mining, do not change tresholds, find a new package and repeat this process.
2. Each miner has perfect knowledge of the problem difficulties (essentially difficulty estimation is perfect)
3. Problem fee is a function of the difficulty and difficulty only
4. Each miner finds packets with rate equal to the miner's compute power (e.g if a miner has computer power equal to 5 then the miner finds 5 packets per second)
## Simulation
### Parameters
The parameters of the main `MiningSimulator` class are:
- `problem_cnt`: The number of problems in the instance pool.
- `packet_size`: The number of problems in a packet.
- `miner_compute_powers`: A numpy array of size (number of miners) that contains the compute power of each miner (in difficulty/s).
- `miner_thresholds_low`: Left bound of the threshold interval for each miner. A numpy array of size (number of miners).
- `miner_thresholds_high`: Right bound of the threshold interval for each miner. A numpy array of size (number of miners).
- `iterations`: Number of iterations to run the simulation for.
- `packet_creator`: Packet creator class (check `PacketCreatorSimulated` in `packet_creation.py` for the interface). Defaults to `PacketCreatorSimulated`.
- `difficulty_generator`: The function that generates the difficulties of the problems in the instance pool. Defaults to `get_difficulties_pareto`.
### Initialization
- Run difficulty generator and store in `self.problem_difficulties` these are the difficulties of problems currently in the "instance pool".
- Compute remaining times (how much time each miner will need for each problem) this is a numpy array of shape (number of miners, number of problems). Store in `remaining_times`.

### Simulation loop
The simulator performs the following steps the given number of times:
- For each miner find the packet that fits the threshold. Compute the time needed to find the packet.
- For each miner compute the time spent to mine the packet (the time that miner would need to create a block).
- Find the winner (miner that has the lowest time spent searching + time spent mining).
- Reward the winner as per the `get_fee` function.
- Reset the difficulties of problems that were mined by the winner. Update `self.problem_difficulties`.
- Update the remaining times for all miners - although the other miners did not mine a block, they still solved some problems, or made progress. Update `remaining_times`.
- For each miner update the remaining times for the new problems that were generated instead of the mined problems. Update `remaining_times`.
