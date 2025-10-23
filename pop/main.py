import numpy as np

from pop import LasserreHierarchy


def example1():
    """
    Detecting Global Optimality and Extracting Solutions in GloptiPoly
    Example in Section 2.3.
    """
    print("GloptiPoly Example in Section 2.3")
    print("min  -2x_1^2 - 2x_2^2 + 2x_1x_2 + 2x_1 + 6x_2 - 10")
    print("s.t. -x_1^2 + 2x_1 >= 0")
    print("     -x_1^2 - x_2^2 + 2x_1x_2 + 1 >=0")
    print("     -x_2^2 + 6x_2 - 8 >=0")
    print()

    f_dict = {(2, 0): -2, (0, 2): -2, (1, 1): 2, (1, 0): 2, (0, 1): 6, (0, 0): -10}
    g_dict_list = [
        {(2, 0): -1, (1, 0): 2},
        {(2, 0): -1, (0, 2): -1, (1, 1): 2, (0, 0): 1},
        {(0, 2): -1, (0, 1): 6, (0, 0): -8},
    ]
    model = LasserreHierarchy(n_vars=2, f_dict=f_dict, g_dict_list=g_dict_list)
    result = model.solve(verbose=True)

    print()
    print(f"value: {np.round(result['value'], 4)}")
    for i, sol in enumerate(result["solutions"]):
        print(f"solution {i + 1}: {np.round(sol, 4)}")


def example2():
    """
    Detecting Global Optimality and Extracting Solutions in GloptiPoly
    Example in Section 3.1.
    """
    print("GloptiPoly Example in Section 3.1")
    print("x_1^2 + x_2^2 - 1 = 0")
    print("x_1^3 + 2x_1x_2 + x_1x_2x_3 + x_2^3 - 1 = 0")
    print("x_3^2 -2 = 0")
    print()

    h_dict_list = [
        {(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 0): -1},
        {(3, 0, 0): 1, (1, 1, 0): 2, (1, 1, 1): 1, (0, 3, 0): 1, (0, 0, 0): -1},
        {(0, 0, 2): 1, (0, 0, 0): -2},
    ]
    model = LasserreHierarchy(n_vars=3, h_dict_list=h_dict_list)
    result = model.solve(verbose=True)

    print()
    print(f"value: {np.round(result['value'], 4)}")
    for i, sol in enumerate(result["solutions"]):
        print(f"solution {i + 1}: {np.round(sol, 4)}")


def example3():
    """
    Detecting Global Optimality and Extracting Solutions in GloptiPoly
    Example in Section 3.2.
    """
    print("GloptiPoly Example in Section 3.2")
    print(" 5x_1^9 - 6x_1^5x_2 + x_1x_2^4 + 2x_1x_3 = 0")
    print("-2x_1^6x_2 + 2x_1^2x_2^3 + 2x_2x_3 = 0")
    print(" x_1^2 + x_2^2 - 0.265625 = 0")
    print()

    h_dict_list = [
        {(9, 0, 0): 5, (5, 1, 0): -6, (1, 4, 0): 1, (1, 0, 1): 2},
        {(6, 1, 0): -2, (2, 3, 0): 2, (0, 1, 1): 2},
        {(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 0): -0.265625},
    ]
    model = LasserreHierarchy(n_vars=3, h_dict_list=h_dict_list)
    result = model.solve(verbose=True)

    print()
    print(f"value: {np.round(result['value'], 4)}")
    for i, sol in enumerate(result["solutions"]):
        print(f"solution {i + 1}: {np.round(sol, 4)}")


def example4():
    """
    Detecting Global Optimality and Extracting Solutions in GloptiPoly
    Example in Section 3.3.
    """
    print("GloptiPoly Example in Section 3.3")
    print("min  -x_1^2 - x_2^2 - x_3^2 +2x_1 + 2x_2 + 2x_3 - 3")
    print("s.t. -x_1^2 + 2x_1 >= 0")
    print("     -x_2^2 + 2x_2 >= 0")
    print("     -x_3^2 + 2x_3 >= 0")
    print()

    f_dict = {
        (2, 0, 0): -1,
        (0, 2, 0): -1,
        (0, 0, 2): -1,
        (1, 0, 0): 2,
        (0, 1, 0): 2,
        (0, 0, 1): 2,
        (0, 0, 0): -3,
    }
    g_dict_list = [
        {(2, 0, 0): -1, (1, 0, 0): 2},
        {(0, 2, 0): -1, (0, 1, 0): 2},
        {(0, 0, 2): -1, (0, 0, 1): 2},
    ]
    model = LasserreHierarchy(n_vars=3, f_dict=f_dict, g_dict_list=g_dict_list)
    result = model.solve(verbose=True)

    print()
    print(f"value: {np.round(result['value'], 4)}")
    for i, sol in enumerate(result["solutions"]):
        print(f"solution {i + 1}: {np.round(sol, 4)}")


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

    example1()

    print("-" * 50)

    example2()

    print("-" * 50)

    example3()

    # print("-" * 50)

    # example4()
