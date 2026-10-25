from city import generate_cities, generate_cost_matrix
from tree_node import TreeNode
from measure_perf import measure_performance
from algorithms.bfs import run_bfs
from algorithms.dfs import run_dfs
from algorithms.nn import run_nn
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar_admissible, run_astar_inadmissible
from algorithms.aco import run_aco
from storage import load_data


def measure_function_random_cities(function, N=6, is_symmetric=True, density=1.0):
    print(f'running {function} with N={N}, is_symmetric={is_symmetric}, density={density}')
    my_cities = generate_cities(N)
    my_matrix = generate_cost_matrix(my_cities, is_symmetric, density)
    path, cost, t, mem = measure_performance(function, my_matrix, N)
    print(f"{function}: Cost {cost:.2f}, Time: {t:.4f} ms, Memory: {mem} B")
    print(path)

def measure_function_file(function, filename):
    print(f'running {function} with file={filename}')
    cities, matrix = load_data(filename)
    N = len(cities)
    path, cost, t, mem = measure_performance(function, matrix, N)
    print(f"{function}: Cost {cost:.2f}, Time: {t:.4f} ms, Memory: {mem} B")
    print(path)


if __name__ == "__main__":
    print("main")
    measure_function_file(run_astar_admissible, "tsp_data_6_symmetric_1_0")
    measure_function_file(run_nn, "tsp_data_5_asymmetric_0_8")
