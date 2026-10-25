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

def tree_test():
    N = 4 # Używamy tylko 4 miast, żeby łatwo było czytać wyniki
    print(f"--- GENEROWANIE ŚRODOWISKA DLA {N} MIAST ---")
    my_cities = generate_cities(N)
    
    # Scenariusz 100% symetryczny
    matrix = generate_cost_matrix(my_cities, is_symmetric=True, density=1.0)
    
    # Wypiszmy macierz dla podglądu
    print("\nMacierz kosztów (odległości):")
    for row in matrix:
        print([f"{val:.1f}" if val != float('inf') else "inf" for val in row])

    print("\n--- BUDOWANIE DRZEWA STANÓW ---")
    
    # KROK 1: Tworzymy korzeń drzewa (Startujemy w mieście 0)
    root_node = TreeNode(current_city=0, path=[0], visited={0}, total_cost=0.0)
    print(f"POZIOM 0 (Korzeń): {root_node}")
    
    # KROK 2: Generujemy dzieci dla korzenia
    level_1_children = root_node.generate_children(matrix, total_cities=N)
    print("\nPOZIOM 1 (Dzieci korzenia - gdzie możemy pojechać z miasta 0?):")
    for child in level_1_children:
        print(f" -> {child}")
        
    # KROK 3: Bierzemy pierwsze z brzegu dziecko (to, które pojechało do miasta 1) i generujemy jego dzieci
    if level_1_children:
        first_child = level_1_children[0]
        level_2_children = first_child.generate_children(matrix, total_cities=N)
        
        print(f"\nPOZIOM 2 (Dzieci węzła, który pojechał z 0 do {first_child.current_city}):")
        for grandchild in level_2_children:
            print(f" -> {grandchild}")

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