import heapq
from tree_node import TreeNode
from city import generate_cities, generate_cost_matrix

def calculate_heuristic(current_city, visited, total_cities, cost_matrix, start_city, is_admissible):

    unvisited = [city for city in range(total_cities) if city not in visited]

    if not unvisited:
        return cost_matrix[current_city][start_city]
        
    if is_admissible:
        # optimistic, shortest possible path to complete the tour
        min_to_unvisited = min(cost_matrix[current_city][u] for u in unvisited)
        min_to_start = min(cost_matrix[u][start_city] for u in unvisited)
        # always lower then the actual cost
        return min_to_unvisited + min_to_start
    else:
        # pesimistic, longest possible path to complete the tour
        max_to_unvisited = max(cost_matrix[current_city][u] for u in unvisited)
        return max_to_unvisited * len(unvisited)


def run_astar(cost_matrix, total_cities, start_city=0, is_admissible=True):
    
    root_node = TreeNode(start_city, [start_city], {start_city}, 0.0)
    # h_cost: heuristic cost, f_cost: total cost (g + h)
    h_cost = calculate_heuristic(start_city, {start_city}, total_cities, cost_matrix, start_city, is_admissible)
    f_cost = root_node.total_cost + h_cost
    
    # priority queue (min-heap), sorts by f_cost and then other
    pq = [(f_cost, root_node.total_cost, id(root_node), root_node)]
    
    best_path = None
    best_cost = float('inf')
    
    while pq:
        # cheapest node
        current_f, current_g, _, current_node = heapq.heappop(pq)
        
        if current_g >= best_cost:
            continue
            
        if len(current_node.visited) == total_cities:
            return_cost = cost_matrix[current_node.current_city][start_city]
            if return_cost != float('inf'):
                total_route_cost = current_node.total_cost + return_cost
                if total_route_cost < best_cost:
                    best_cost = total_route_cost
                    best_path = current_node.path + [start_city]
            continue
            
        children = current_node.generate_children(cost_matrix, total_cities)
        
        for child in children:
            if child.total_cost < best_cost:
                h = calculate_heuristic(
                    child.current_city, child.visited, total_cities, 
                    cost_matrix, start_city, is_admissible
                )
                f = child.total_cost + h
                
                heapq.heappush(pq, (f, child.total_cost, id(child), child))
                
    return best_path, best_cost

def run_astar_admissible(cost_matrix, total_cities, start_city=0):
    return run_astar(cost_matrix, total_cities, start_city, is_admissible=True)

def run_astar_inadmissible(cost_matrix, total_cities, start_city=0):
    return run_astar(cost_matrix, total_cities, start_city, is_admissible=False)


if __name__ == "__main__":
    N = 6
    my_cities = generate_cities(N)
    my_matrix = generate_cost_matrix(my_cities, is_symmetric=True, density=1.0)
    path_adm, cost_adm = run_astar_admissible(my_matrix, total_cities=N, start_city=0)
    print(f"route: {path_adm}, cost: {cost_adm:.2f}")
    path_inadm, cost_inadm = run_astar_inadmissible(my_matrix, total_cities=N, start_city=0)
    print(f"route: {path_inadm}, cost: {cost_inadm:.2f}")