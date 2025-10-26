from itertools import combinations_with_replacement


class BasisUtils:
    def __init__(self, num_vars):
        self.num_vars = num_vars

    def _generate_basis(self, degree):
        """
        Generate monomial basis in terms of tuples.
        Ex. [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]
        ->  [(1, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        """
        basis = []
        for d in range(degree + 1):
            for exp in combinations_with_replacement(range(self.num_vars), d):
                exponent = [0] * self.num_vars
                for var in exp:
                    exponent[var] += 1
                basis.append(tuple(exponent))
        return basis

    def _multiply_basis(self, basis1, basis2):
        """
        Monomial basis multiplication.
        Ex. x_1^2 @ x1x2 = x_1^3x_2
        ->  (2, 0) + (1, 1) = (3, 1)
        """
        return tuple(basis1[i] + basis2[i] for i in range(self.num_vars))
