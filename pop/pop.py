import math
import numpy as np
import cvxpy as cp
from scipy.linalg import schur
import sympy as sp

from utils import Basis


class LasserreHierarchy(Basis):
    def __init__(self, n_vars, f_dict=None, g_dict_list=None, h_dict_list=None):
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

        constraints = []
        constraints.append(y[0] == 1)

        # moment matrix
        M, M_constraints = self.build_moment_matrix(y)
        constraints.append(M >> 0)
        constraints.extend(M_constraints)

        if self.f_dict is not None:
            c = self.poly_dict_to_vector(self.f_dict)
            objective = cp.Minimize(c @ y)
        else:
            objective = cp.Minimize(cp.trace(M))

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
            status = problem.status == cp.OPTIMAL
            moment_matrix = M.value
        except:
            status = False
            moment_matrix = None

        return {
            "status": status,
            "value": problem.value,
            "moment_matrix": moment_matrix,
        }

    def extract_solutions(self, M):
        solutions = []
        eigvals, eigvecs = np.linalg.eigh(M)
        rank = np.sum(eigvals > 1e-3)

        if rank == 1:
            v = M[:, 0]  # first column
            x = np.zeros(self.n_vars)
            for i in range(self.n_vars):
                basis_elem = tuple([1 if j == i else 0 for j in range(self.n_vars)])
                idx = self.basis_kappa.index(basis_elem)
                x[i] = v[idx]
            solutions.append(x)
        else:
            # cholesky factor
            sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
            V_full = eigvecs @ np.diag(sqrt_eigvals)
            V = V_full[:, eigvals > 1e-3]

            # column echelon form
            Vt_sp = sp.Matrix(V.T)
            rref_matrix, pivots = Vt_sp.rref()
            U = np.array(rref_matrix, dtype=float).T
            # U = np.round(U)

            # basis corresponding to pivots
            w_basis = [self.basis_kappa[i] for i in pivots]

            # multiplication matrices x_i * w(x) = M_i @ w(x)
            multi_matrices = []
            for var_idx in range(self.n_vars):
                N_i = np.zeros((rank, rank))
                for j, w in enumerate(w_basis):
                    xw_list = list(w)  # w_j
                    xw_list[var_idx] += 1  # x_i * w_j
                    xw = tuple(xw_list)
                    N_i[j, :] = U[self.basis_2kappa.index(xw), :]
                multi_matrices.append(N_i)

            # convex combination
            coeffs = np.random.rand(self.n_vars)
            coeffs /= np.sum(coeffs)
            N = sum(coeff * N_i for coeff, N_i in zip(coeffs, multi_matrices))

            # extract solutions
            T, Q = schur(N, output="real")
            for j in range(rank):
                x = np.zeros(self.n_vars)
                q_j = Q[:, j]
                for i, N_i in enumerate(multi_matrices):
                    x[i] = q_j.T @ N_i @ q_j
                solutions.append(x)

        return solutions

    def solve(self, verbose=False):
        v = 0
        if self.g_dict_list is not None:
            for g_dict in self.g_dict_list:
                max_deg_g = max(sum(exp) for exp in g_dict.keys())
                v = max(v, math.ceil(max_deg_g / 2))
        if self.h_dict_list is not None:
            for h_dict in self.h_dict_list:
                max_deg_h = max(sum(exp) for exp in h_dict.keys())
                v = max(v, math.ceil(max_deg_h / 2))

        if self.f_dict is not None:
            max_deg_f = max(sum(exp) for exp in self.f_dict.keys())
        else:
            max_deg_f = 0
            if verbose:
                print("[POP] No objective found, minimize the trace of moment matrix")
        kappa = max(v, max_deg_f)

        if verbose:
            print(f"[POP] rank diff condition: {v}", end="\n\n")

        for i in range(10):  # ! hard code 10 hierarchy levels
            if verbose:
                print(f"[POP] {kappa}-th order moment relaxation:")

            result = self.solve_moment_relaxation(kappa)

            if verbose:
                print(f"[POP] sdp optimality: {result['status']}")

            if result["status"]:
                M = result["moment_matrix"]

                # check flat extension
                is_global = False
                v_counter = 0
                for i in range(kappa):
                    if not i:
                        n_prev = len(self._generate_basis(i + 1))
                        M_prev = M[:n_prev, :n_prev]
                        eigvals_sv = np.linalg.eigvalsh(M_prev)
                        prev_rank = np.sum(eigvals_sv > 1e-3)
                        if verbose:
                            print(f"[POP] rank(M_{i + 1})={prev_rank}")
                    else:
                        n_curr = len(self._generate_basis(i + 1))
                        M_curr = M[:n_curr, :n_curr]
                        eigvals_sv = np.linalg.eigvalsh(M_curr)
                        curr_rank = np.sum(eigvals_sv > 1e-3)
                        if verbose:
                            print(f"[POP] rank(M_{i + 1})={curr_rank}")

                        if curr_rank == prev_rank:
                            v_counter += 1
                            if v_counter >= v:
                                is_global = True
                                M_extract = M_curr
                                break
                        else:
                            v_counter = 0
                        prev_rank = curr_rank

                if verbose:
                    print(f"[POP] pop optimality: {is_global}")
                if is_global:
                    solutions = self.extract_solutions(M_extract)
                    return {
                        "value": result["value"],
                        "solutions": solutions,
                    }

            kappa += 1
            if verbose:
                print()

        raise RuntimeError("No solution found")


if __name__ == "__main__":
    print("min x^2 s.t. x >= 2")
    f_dict = {(2,): 1}
    g_dict_list = [{(1,): 1, (0,): -2}]
    model = LasserreHierarchy(n_vars=1, f_dict=f_dict, g_dict_list=g_dict_list)
    result = model.solve()
    print(f"val: {result['value']:.1f}, sol: {result['solutions'][0][0]:.1f}")
