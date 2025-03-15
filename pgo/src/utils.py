import numpy as np

""" 
[1] Sola, J., Deray, J., & Atchuthan, D. (2018).
    A micro Lie theory for state estimation in robotics.
    arXiv preprint arXiv:1812.01537. 
"""
APPROX_EPS = 1e-10  # handle small theta


def se2_mat_to_vec(mat):
    x = mat[0, 2]
    y = mat[1, 2]
    theta = np.arctan2(mat[1, 0], mat[0, 0])
    return np.array([x, y, theta])


def se2_vec_to_mat(vec):
    theta = vec[2]
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mat = np.eye(3)
    mat[0:2, 0:2] = rot
    mat[:2, 2] = vec[:2]
    return mat


def se2_Exp(vec):
    """Maps Cartesian vector to SE(2) matrix [1, Eq. (156)]."""
    theta = vec[2]
    theta_sq = theta * theta

    if theta_sq < APPROX_EPS:  # Taylor approximation
        A = 1.0 - theta_sq / 6.0
        B = 0.5 * theta - theta * theta_sq / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta

    V = np.array([[A, -B], [B, A]])
    t = V @ vec[:2]

    mat = np.eye(3)
    mat[:2, :2] = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    mat[:2, 2] = t
    return mat


def se2_Log(mat):
    """Maps SE(2) matrix to Cartesian vector [1, Eq. (157)]."""
    theta = np.arctan2(mat[1, 0], mat[0, 0])
    theta_sq = theta * theta

    if theta_sq < APPROX_EPS:  # Taylor approximation
        A = 1.0 - theta_sq / 6.0
        B = 0.5 * theta - theta * theta_sq / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta

    det = A * A + B * B
    V_inv = np.array([[A, B], [-B, A]]) / det
    rho = V_inv @ mat[:2, 2]
    return np.array([rho[0], rho[1], theta])


def se2_Adjoint(mat):
    """Adjoint matrix of SE(2) matrix [1, Eq. (159)]."""
    rot = mat[:2, :2]
    t = mat[:2, 2]

    adj = np.eye(3)
    adj[:2, :2] = rot
    adj[:2, 2] = np.array([t[1], -t[0]])
    return adj


def se2_Jacobian_right(vec):
    """Right Jacobian of Cartesian vector [1, Eq. (163)]."""
    theta = vec[2]
    theta_sq = theta * theta

    if theta_sq < APPROX_EPS:  # Taylor approximation
        A = 1.0 - theta_sq / 6.0
        B = 0.5 * theta - theta * theta_sq / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta

    J_r = np.eye(3)
    J_r[:2, :2] = np.array([[A, -B], [B, A]])

    if theta_sq < APPROX_EPS:  # Taylor approximation
        J_r[0, 2] = -vec[1] / 2.0 + theta * vec[0] / 6.0
        J_r[1, 2] = vec[0] / 2.0 + theta * vec[1] / 6.0
    else:
        J_r[0, 2] = (
            -vec[1] + theta * vec[0] + vec[1] * np.cos(theta) - vec[0] * np.sin(theta)
        ) / theta_sq
        J_r[1, 2] = (
            vec[0] + theta * vec[1] - vec[0] * np.cos(theta) - vec[1] * np.sin(theta)
        ) / theta_sq

    return J_r
