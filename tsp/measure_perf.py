import time
import tracemalloc


def measure_performance(algorithm_function, cost_matrix, total_cities, start_city=0):

    tracemalloc.start()
    start_time = time.perf_counter()

    best_path, best_cost = algorithm_function(cost_matrix, total_cities, start_city)

    exec_time_ms = (time.perf_counter() - start_time) * 1000
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return best_path, best_cost, exec_time_ms, peak_memory_bytes

if __name__ == "__main__":
    print("measure performance")