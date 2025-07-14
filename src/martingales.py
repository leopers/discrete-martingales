import numpy as np


def simulate_martingale_betting_strategy(n_paths=1000, n_steps=20, seed=None):
    """
    Simulate a doubling martingale betting strategy:
    - Bets 1 at first step
    - Doubles bet each time there's a loss (X_j = -1)
    - Stops betting after first win (X_j = +1)

    Parameters:
    - n_paths: Number of sample paths
    - n_steps: Maximum number of steps
    - seed: Random seed (optional)

    Returns:
    - X: coin tosses (n_paths, n_steps)
    - B: bets placed at each step (n_paths, n_steps)
    - W: cumulative winnings (n_paths, n_steps)
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.choice([-1, 1], size=(n_paths, n_steps))  # ±1 tosses
    B = np.zeros_like(X)
    W = np.zeros_like(X)

    for i in range(n_paths):
        betting = True
        for t in range(n_steps):
            if not betting:
                B[i, t] = 0
                W[i, t] = W[i, t - 1]
                continue

            # Doubling strategy: B_1 = 1, then double after each loss
            if t == 0:
                B[i, t] = 1
            elif X[i, t - 1] == -1:
                B[i, t] = 2 * B[i, t - 1]
            else:
                B[i, t] = 0
                betting = False  # stop betting after first win

            # Compute winnings
            delta = B[i, t] * X[i, t]
            W[i, t] = W[i, t - 1] + delta if t > 0 else delta

    return X, B, W


def check_martingale_property_by_time(process, tol=1e-6):
    """
    Check whether E[X_{t+1}] ≈ E[X_t] for each time t.
    Does not condition on filtration — only checks global mean.

    Parameters:
    - process: array (n_paths, n_steps)
    - tol: acceptable deviation

    Returns:
    - List[bool]: whether the martingale condition holds at each time t
    """
    n_paths, n_steps = process.shape
    diffs = []
    for t in range(n_steps - 1):
        mean_prev = np.mean(process[:, t])
        mean_next = np.mean(process[:, t + 1])
        diffs.append(abs(mean_next - mean_prev) < tol)
    return diffs
