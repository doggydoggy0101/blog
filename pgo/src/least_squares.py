import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, onenormest

from src.utils import (
    se2_vec_to_mat,
    se2_mat_to_vec,
    se2_Log,
    se2_Exp,
    se2_Jacobian_right,
    se2_Adjoint,
)


class GaussNewton:
    def __init__(self, max_iteration, tolerance):
        self.max_iteration = max_iteration
        self.tolerance = tolerance

    def compute_Jacobian_and_residual(self, node1, node2, edge, gradType):
        pose_i = se2_vec_to_mat(node1)
        pose_j = se2_vec_to_mat(node2)
        pose_ij = se2_vec_to_mat(edge)
        rel_pose = np.linalg.inv(pose_ij) @ np.linalg.inv(pose_i) @ pose_j

        if gradType == "Euclidean":
            # linearized residual
            res = se2_mat_to_vec(rel_pose)

            rot_i = pose_i[:2, :2]
            rot_ij = pose_ij[:2, :2]

            # derivative of rot_i.T with respect to theta_i
            theta_i = node1[2]
            drotT_i = np.array(
                [
                    [-np.sin(theta_i), np.cos(theta_i)],
                    [-np.cos(theta_i), -np.sin(theta_i)],
                ]
            )

            # Jacobian of residual with respect to pose_i
            J_i = -np.eye(3)
            J_i[:2, :2] = -rot_ij.T @ rot_i.T
            J_i[:2, 2] = rot_ij.T @ drotT_i @ (node2[:2] - node1[:2])

            # Jacobian of residual with respect to pose_j
            J_j = np.eye(3)
            J_j[:2, :2] = rot_ij.T @ rot_i.T

        elif gradType == "Lie":
            # linearized residual
            res = se2_Log(rel_pose)

            Jr_inv = np.linalg.inv(se2_Jacobian_right(res))

            # Jacobian of residual with respect to pose_j
            J_j = Jr_inv

            # Jacobian of residual with respect to pose_i
            J_i = -Jr_inv @ se2_Adjoint(np.linalg.inv(pose_j) @ pose_i)

        else:
            print("gradType not defined (see `src/least_squares.py`)")
            exit(1)

        return J_i, J_j, res

    def compute_increment(self, graph, gradType):
        n = graph.x.shape[0]
        H = np.zeros((n, n))
        b = np.zeros(n)

        addPrior = True

        for edge in graph.edges:
            i = graph.lut[edge.fromNode]
            j = graph.lut[edge.toNode]

            pose_i = graph.x[i : i + 3]
            pose_j = graph.x[j : j + 3]
            constraint = edge.measurement
            J_i, J_j, res = self.compute_Jacobian_and_residual(
                pose_i, pose_j, constraint, gradType
            )

            mat_info = edge.information
            H[i : i + 3, i : i + 3] += J_i.T @ mat_info @ J_i
            H[i : i + 3, j : j + 3] += J_i.T @ mat_info @ J_j
            H[j : j + 3, i : i + 3] += J_j.T @ mat_info @ J_i
            H[j : j + 3, j : j + 3] += J_j.T @ mat_info @ J_j
            b[i : i + 3] += J_i.T @ mat_info @ res
            b[j : j + 3] += J_j.T @ mat_info @ res

            if addPrior:
                H[i : i + 3, i : i + 3] += 1000 * np.eye(3)
                addPrior = False

        # TODO ill-condition
        # H_sparse = csc_matrix(H)
        # H_cond = onenormest(H_sparse)
        # if H_cond > 1e+10:
        #     H += 1e-3 * np.eye(n)

        H_sparse = csc_matrix(H)
        H_factorized = splu(H_sparse)
        dx = H_factorized.solve(-b)

        return dx

    def solve(self, graph, gradType, verbose=True):
        if verbose:
            print(f"Solving with gradType: {gradType}")
        for i in range(self.max_iteration):
            dx = self.compute_increment(graph, gradType)
            norm_dx = np.linalg.norm(dx)
            if verbose:
                print(f"Iteration {i + 1}, norm_dx: {norm_dx}")

            if gradType == "Euclidean":
                graph.x += dx
            elif gradType == "Lie":
                for idx in range(0, len(graph.x), 3):
                    pose_vec = graph.x[idx : idx + 3]
                    dx_vec = dx[idx : idx + 3]
                    updated_pose = se2_mat_to_vec(
                        se2_vec_to_mat(pose_vec) @ se2_Exp(dx_vec)
                    )
                    graph.x[idx : idx + 3] = updated_pose

            # stopping criteria
            if norm_dx < self.tolerance:
                if verbose:
                    print(f"Converged in {i + 1} iterations", end="\n\n")
                break

        return graph
