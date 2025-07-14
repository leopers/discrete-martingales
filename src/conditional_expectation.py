import numpy as np


def simulate_coin_tosses(n_paths=1000, n_steps=10, seed=None):
    """
    Simulate n_paths of n_steps coin tosses.

    Parameters:
    - n_paths: Number of paths to simulate.
    - n_steps: Number of steps in each path.
    - seed: Random seed for reproducibility.

    Returns:
    - A 2D numpy array of shape (n_paths, n_steps) with simulated coin tosses.
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.choice([-1, 1], size=(n_paths, n_steps))


def build_partial_sums(coin_tosses):
    """
    Build partial sums from coin tosses.

    Parameters:
    - coin_tosses: A 2D numpy array of shape (n_paths, n_steps) with coin tosses.

    Returns:
    - A 2D numpy array of shape (n_paths, n_steps) with partial sums.
    """
    return np.cumsum(coin_tosses, axis=1)


def estimate_conditional_expectation(coin_tosses, t):
    """
    Estimate the conditional expectation of the partial sums at time t.

    Parameters:
    - coin_tosses: A 2D numpy array of shape (n_paths, n_steps) with coin tosses.
    - t: Time step at which to estimate the conditional expectation.

    Returns:
    - Mapping history tuples to their estimated conditional expectations.
    """

    n_paths, n_steps = coin_tosses.shape
    if not (0 <= t < n_steps - 1):
        raise ValueError("t must satisfy 0 ≤ t < n_steps−1")
    groups = {}
    for path in coin_tosses:
        history = tuple(path[: t + 1])
        nxt = path[t + 1]
        groups.setdefault(history, []).append(nxt)
    return {h: np.mean(v) for h, v in groups.items()}


def estimate_conditional_expectation_sums(partial_sums, t):
    """
    Estimate the conditional expectation of the partial sums at time t.

    Parameters:
    - partial_sums: A 2D numpy array of shape (n_paths, n_steps) with partial sums.
    - t: Time step at which to estimate the conditional expectation.

    Returns:
    - Mapping history tuples to their estimated conditional expectations.
    """

    n_paths, n_steps = partial_sums.shape
    if not (0 <= t < n_steps - 1):
        raise ValueError("t must satisfy 0 ≤ t < n_steps−1")
    groups = {}
    for S in partial_sums:
        st = S[t]
        snext = S[t + 1]
        groups.setdefault(st, []).append(snext)
    return {s: np.mean(v) for s, v in groups.items()}
