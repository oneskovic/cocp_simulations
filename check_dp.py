import numpy as np
from tqdm import tqdm
rounding_interval = 0.2
EPS = 0.0001

# Check all subsets of size k
def __bruteforce(elements, k, low, high, pos, curr_set):
    if pos == len(elements):
        if len(curr_set) == k:
            curr_sum = sum(curr_set)
            if low - EPS <= curr_sum <= high + EPS:
                return 1
        return 0
    
    curr_set.append(elements[pos])
    cnt_with = __bruteforce(elements, k, low, high, pos+1, curr_set)
    curr_set.pop()
    cnt_without = __bruteforce(elements, k, low, high, pos+1, curr_set)
    return cnt_with + cnt_without

# Count number of subsets with exactly k elements with sum between low and high
def bruteforce(elements, k, low, high):
    return __bruteforce(elements, k, low, high, 0, [])

def solve_dp(elements, set_size, low, high, rounding_interval):
    # Map elements to ints in [0,1,...]
    elements = [int(round(x/rounding_interval)) for x in elements]
    low = int(round(low/rounding_interval))
    high = int(round(high/rounding_interval))

    n = len(elements)
    max_sum = sum(sorted(elements)[-set_size:])
    dp = np.zeros((n+1,set_size+1,max_sum+1), dtype=np.int32)
    for i in range(1,n+1):
        # Don't take element i (faster than to calculate in loop)
        dp[i] = dp[i-1].copy()
        # Take just element i
        dp[i,1,elements[i-1]] += 1
        for k in range(1,set_size+1):
            for s in range(max_sum+1):
                if s - elements[i-1] >= 0:
                    # Take element i -> number of sets with k-1 elements and sum s-elements[i-1]
                    dp[i,k,s] += dp[i-1,k-1,s-elements[i-1]]

    sol = dp[-1][-1][low:high+1].sum()
    return sol

def test():
    n_tests = 1000
    elem_min = 0
    elem_max = 50
    max_n = 11
    max_k = 5

    cnt_nonzero = 0
    pbar = tqdm(total=n_tests)
    for _ in range(n_tests):
        n = np.random.randint(1,max_n+1)
        k = np.random.randint(1,min(n,max_k)+1)
        elem_range = np.arange(elem_min, elem_max, rounding_interval)
        elements = np.random.choice(elem_range, n)
        low,high = sorted(np.random.choice(elem_range, 2, replace=False))

        sol1 = bruteforce(elements, k, low, high)
        sol2 = solve_dp(elements, k, low, high, rounding_interval)
        if sol1 != 0:
            cnt_nonzero += 1
        pbar.update(1)
        pbar.set_description(f'Nonzero: {cnt_nonzero}')
        assert sol1 == sol2
    pbar.close()

test()

# n,k = [int(x) for x in input().split()]
# low, high = [float(x) for x in input().split()]
# elements = [float(x) for x in input().split()]
# print(bruteforce(elements, k, low, high))
# print(solve_dp(elements, k, low, high, rounding_interval))