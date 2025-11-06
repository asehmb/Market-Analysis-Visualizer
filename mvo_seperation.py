import numpy as np
import matplotlib.pyplot as plt

# Accounts for riskfree holdings
# does not exclude shortselling

def feasibility(m, rf, e):
    right_side = rf*e
    for i in range(m.shape[0]):
        if right_side[i] != m[i]:
            return True
    return False

def riskFreeSolution(z,m,rf,e, expected_return):

    sigma = np.cov(z, rowvar=False)
    sigma_inv = np.linalg.inv(sigma)

    # denominator of alpha
    denominator = (m - rf * e).T @ sigma_inv @ (m - rf * e)
    alpha = (expected_return - rf) / denominator

    w_p_tangency = sigma_inv @ (m - rf * e) # bottom of vector

    w_b_tangency = 1 - e.T @ w_p_tangency  # top of vector

    # w = (1-alpha) * w_1 + alpha * w_2

    w_b_final = (1 - alpha) * 1.0 + alpha * w_b_tangency # w0

    w_p_final = (1 - alpha) * 0.0 + alpha * w_p_tangency # w


    weights = np.hstack([w_b_final, w_p_final])

    return weights

if __name__ == "__main__":
    # 3 assets 5 time periods
    z = np.array([
        [0.05, 0.02, 0.03],
        [0.04, 0.04, 0.02],
        [0.06, 0.03, 0.04],
        [0.07, 0.02, 0.05],
        [0.03, 0.02, 0.03]
    ])

    m = np.mean(z, axis=0)

    e = np.ones(m.shape[0])

    expected_return = 0.05
    risk_free_rate = 0.03

    is_feasible = feasibility(m, risk_free_rate, e)
    print(is_feasible)
    weights = riskFreeSolution(z,m,risk_free_rate,e,expected_return)
    print(weights)
