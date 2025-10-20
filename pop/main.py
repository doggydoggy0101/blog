import math
import numpy as np
import cvxpy as cp
from scipy.linalg import qr, schur

from utils import Basis


class LasserreHierarchy(Basis):
    def __init__(self, n_vars, f_dict, g_dict_list=None, h_dict_list=None):
        super().__init__(n_vars)

        self.n_vars = n_vars
        self.f_dict = f_dict
        self.g_dict_list = g_dict_list
        self.h_dict_list = h_dict_list

    def poly_dict_to_vector(self, poly_dict):
        y_coeffs = np.zeros(self.n_2kappa)
        for basis, coeff in poly_dict.items():
            idx = self.basis_2kappa.index(basis)
            y_coeffs[idx] = coeff
        return y_coeffs

    def build_moment_matrix(self, y):
        M = cp.Variable((self.n_kappa, self.n_kappa), symmetric=True)
        constraints = []
        # M is symmetric, constraint only on upper triangular
        for i in range(self.n_kappa):
            for j in range(i, self.n_kappa):
                product = self._multiply_basis(self.basis_kappa[i], self.basis_kappa[j])
                y_idx = self.basis_2kappa.index(product)
                constraints.append(M[i, j] == y[y_idx])

        return M, constraints

    def build_localizing_matrix(self, y, poly_dict, deg):
        basis_local = self._generate_basis(deg)
        n_local = len(basis_local)

        M = cp.Variable((n_local, n_local), symmetric=True)
        constraints = []
        # M is symmetric, constraint only on upper triangular
        for i in range(n_local):
            for j in range(i, n_local):
                basis_product = self._multiply_basis(basis_local[i], basis_local[j])

                coeff_sum = 0
                for basis, coeff in poly_dict.items():
                    product = self._multiply_basis(basis_product, basis)
                    y_idx = self.basis_2kappa.index(product)
                    coeff_sum += coeff * y[y_idx]

                constraints.append(M[i, j] == coeff_sum)

        return M, constraints

    def solve_moment_relaxation(self, kappa):
        # update parameters
        self.kappa = kappa
        self.basis_kappa = self._generate_basis(kappa)
        self.basis_2kappa = self._generate_basis(2 * kappa)
        self.n_kappa = len(self.basis_kappa)
        self.n_2kappa = len(self.basis_2kappa)

        # moment vector y
        y = cp.Variable(self.n_2kappa)

        # objective
        c = self.poly_dict_to_vector(self.f_dict)
        objective = cp.Minimize(c @ y)

        constraints = []
        constraints.append(y[0] == 1)

        # moment matrix
        M, M_constraints = self.build_moment_matrix(y)
        constraints.append(M >> 0)
        constraints.extend(M_constraints)

        # localizing matrix
        if self.g_dict_list is not None:
            for g_dict in self.g_dict_list:
                max_deg_g = max(sum(exp) for exp in g_dict.keys())
                deg = self.kappa - math.ceil(max_deg_g / 2)
                Mg, Mg_constraints = self.build_localizing_matrix(y, g_dict, deg)
                constraints.append(Mg >> 0)
                constraints.extend(Mg_constraints)

        if self.h_dict_list is not None:
            for h_dict in self.h_dict_list:
                max_deg_h = max(sum(exp) for exp in h_dict.keys())
                deg = self.kappa - math.ceil(max_deg_h / 2)
                Mh, Mh_constraints = self.build_localizing_matrix(y, h_dict, deg)
                constraints.append(Mh == 0)
                constraints.extend(Mh_constraints)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.MOSEK)
            status = problem.status
            moment_matrix = M.value
        except Exception as e:
            print(f"  MOSEK failed: {e}")
            status = "failed"
            moment_matrix = None

        return {
            "status": status,
            "moment_matrix": moment_matrix,
        }

    def extract_solutions(self, M):
        solutions = []
        eigvals, eigvecs = np.linalg.eigh(M)
        rank = np.sum(eigvals > 1e-6)

        if rank == 1:
            v = eigvecs[:, -1]
            v /= v[0]
            x_opt = np.zeros(self.n_vars)
            for i in range(self.n_vars):
                basis_elem = tuple([1 if j == i else 0 for j in range(self.n_vars)])
                idx = self.basis_kappa.index(basis_elem)
                x_opt[i] = v[idx]
            solutions.append(x_opt)
        else:
            # factorization
            sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
            V_full = eigvecs @ np.diag(sqrt_eigvals)
            V = V_full[:, eigvals > 1e-6]

            # pivot points and corresponding basis
            Q, R, P = qr(V, mode="economic", pivoting=True)
            pivot_indices = []
            for i in range(min(R.shape)):
                if abs(R[i, i]) > 0:
                    pivot_indices.append(P[i])
            pivot_indices = sorted(pivot_indices)
            w_basis = [self.basis_kappa[i] for i in pivot_indices]

            # echelon form
            V_reduced = V[pivot_indices, :]
            U = V @ np.linalg.pinv(V_reduced)
            U = np.round(U)

            # multiplication matrices x_i * w(x) = M_i @ w(x)
            multi_matrices = []
            for var_idx in range(self.n_vars):
                N_i = np.zeros((rank, rank))
                for j, w in enumerate(w_basis):
                    xw_list = list(w)  # w_j
                    xw_list[var_idx] += 1  # x_i * w_j
                    xw = tuple(xw_list)
                    N_i[:, j] = U[self.basis_kappa.index(xw), :]
                multi_matrices.append(N_i)

            # convex combination
            coeffs = np.random.rand(self.n_vars)
            coeffs /= np.sum(coeffs)
            N = sum(coeff * M for coeff, M in zip(coeffs, multi_matrices))

            # extract solutions
            T, Q = schur(N, output="real")
            for i in range(rank):
                x = np.zeros(self.n_vars)
                for var_idx in range(self.n_vars):
                    x[var_idx] = Q[:, i].T @ multi_matrices[var_idx] @ Q[:, i]
                solutions.append(x)

        return solutions

    def solve(self):
        v = 0
        if self.g_dict_list is not None:
            for g_dict in self.g_dict_list:
                max_deg_g = max(sum(exp) for exp in g_dict.keys())
                v = max(v, math.ceil(max_deg_g / 2))
        if self.h_dict_list is not None:
            for h_dict in self.h_dict_list:
                max_deg_h = max(sum(exp) for exp in h_dict.keys())
                v = max(v, math.ceil(max_deg_h / 2))

        max_deg_f = max(sum(exp) for exp in f_dict.keys())
        kappa = max(v, max_deg_f)

        # NOTE no example requires heirarchy, not sure if this is implemented correctly
        for i in range(10):  # ! hard code 10 hierarchy levels
            result = self.solve_moment_relaxation(kappa)
            print(f"\n{kappa}-th order moment relaxation status: {result['status']}")

            if result["status"] == "optimal":
                M = result["moment_matrix"]

                eigvals = np.linalg.eigvalsh(M)
                curr_rank = np.sum(eigvals > 1e-6)

                n_prev = len(self._generate_basis(max(kappa - v, 1)))
                M_prev = M[:n_prev, :n_prev]
                eigvals_sv = np.linalg.eigvalsh(M_prev)
                prev_rank = np.sum(eigvals_sv > 1e-6)

                # check flat extension
                is_global = curr_rank == prev_rank
                print(f"Global optimality: {is_global}, rank: {curr_rank}")
                if is_global:
                    solutions = self.extract_solutions(M)
                    return solutions

            kappa += 1

        print("No solution found")
        return []


if __name__ == "__main__":
    # NOTE M4 problem, see # https://github.com/numpy/numpy/issues/29820
    import platform, subprocess, warnings

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip() == "Apple M4":
            for message in [
                "overflow encountered in matmul",
                "divide by zero encountered in matmul",
                "invalid value encountered in matmul",
            ]:
                warnings.filterwarnings("ignore", message=message)

    """
    Detecting Global Optimality and Extracting Solutions in GloptiPoly
    Example in Section 2.3.
    """
    print("Example 1")
    print("min  -2x_1^2 - 2x_2^2 + 2x_1x_2 + 2x_1 + 6x_2 - 10")
    print("s.t. -x_1^2 + 2x_1 >= 0")
    print("     -x_1^2 - x_2^2 + 2x_1x_2 + 1 >=0")
    print("     -x_2^2 + 6x_2 - 8 >=0")

    f_dict = {(2, 0): -2, (0, 2): -2, (1, 1): 2, (1, 0): 2, (0, 1): 6, (0, 0): -10}
    g_dict_list = [
        {(2, 0): -1, (1, 0): 2},
        {(2, 0): -1, (0, 2): -1, (1, 1): 2, (0, 0): 1},
        {(0, 2): -1, (0, 1): 6, (0, 0): -8},
    ]
    model = LasserreHierarchy(n_vars=2, f_dict=f_dict, g_dict_list=g_dict_list)
    solutions = model.solve()
    for i, sol in enumerate(solutions):
        print(f"Solution {i + 1}: {sol}")

    """
    Detecting Global Optimality and Extracting Solutions in GloptiPoly
    Example in Section 3.1.
    """
    print("\nExample 2")
    print("min 0 ")
    print("s.t.  x_1^2 + x_2^2 - 1 = 0")
    print("      x_1^3 + 2x_1x_2 + x_1x_2x_3 + x_2^3 - 1 = 0")
    print("      x_3^2 -2 = 0")

    f_dict = {(0, 0, 0): 0}
    h_dict_list = [
        {(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 0): -1},
        {(3, 0, 0): 1, (1, 1, 0): 2, (1, 1, 1): 1, (0, 3, 0): 1, (0, 0, 0): -1},
        {(0, 0, 2): 1, (0, 0, 0): -2},
    ]
    model = LasserreHierarchy(n_vars=3, f_dict=f_dict, h_dict_list=h_dict_list)
    solutions = model.solve()
    for i, sol in enumerate(solutions):
        print(f"Solution {i + 1}: {sol}")
