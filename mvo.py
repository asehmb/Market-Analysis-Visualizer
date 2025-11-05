import numpy as np
import matplotlib.pyplot as plt


def feasibility(m, target_return):
    """
    Check if a target return is feasible.
    Returns True if the target is between min and max asset means.
    """
    min_return = np.min(m)
    max_return = np.max(m)

    # Allow a small tolerance for floating point comparisons
    if target_return < min_return - 1e-9 or target_return > max_return + 1e-9:
        return False
    return True


def minimum_variance_solution(z, e):
    """
    Calculates the Global Minimum Variance (GMV) portfolio weights.
    """
    sigma = np.cov(z, rowvar=False)
    sigma_inv = np.linalg.inv(sigma)

    # Calculate weights
    w_mv = (sigma_inv @ e) / (e.T @ sigma_inv @ e)
    return w_mv


def two_portfolio_solution(z, m, e, w_mv, expected_return):
    """
    Calculates the efficient portfolio for a target return.
    """
    sigma = np.cov(z, rowvar=False)
    sigma_inv = np.linalg.inv(sigma)

    w_mk = (sigma_inv @ m) / (e.T @ sigma_inv @ m)

    # Define the vector pointing from w_mv to w_mk
    v = w_mk - w_mv

    # Calculate the weighting (alpha) needed to achieve the target return
    alpha = (expected_return - (m @ w_mv)) / (m @ v)

    # The final weights are a combination of the two portfolios
    w = w_mv + alpha * v

    return w


if __name__ == "__main__":
    z = np.array([
        [0.05, 0.02, 0.03],
        [0.04, 0.01, 0.02],
        [0.06, 0.03, 0.04],
        [0.07, 0.02, 0.05],
        [0.05, 0.01, 0.03]
    ])

    # m is (3,)
    m = np.mean(z, axis=0)

    e = np.ones(m.shape[0])

    expected_return = 0.05

    is_feasible = feasibility(m, expected_return)
    print(f"Is target return {expected_return} feasible? {is_feasible}")

    if not is_feasible:
        pass
    else:
        w_mv = minimum_variance_solution(z, e)
        return_mv = w_mv @ m

        print(f"Global Min-Variance portfolio return: {return_mv:.4f}")

        if return_mv >= expected_return:
            print("Solution is the Global Minimum Variance portfolio.")
            solution = w_mv
        else:
            print("Solution is a combination of two efficient portfolios.")
            solution = two_portfolio_solution(z, m, e, w_mv, expected_return)

        print(f"\nOptimal Weights: {solution}")
        print(f"Portfolio Return: {solution @ m:.4f}")
        print(f"Sum of Weights: {np.sum(solution):.4f}")