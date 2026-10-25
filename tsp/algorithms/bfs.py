from tree_node import TreeNode
from city import generate_cities, generate_cost_matrix

def run_bfs(cost_matrix, total_cities, start_city=0):
    root_node = TreeNode(
        current_city=start_city, 
        path=[start_city], 
        visited={start_city}, 
        total_cost=0.0
    )
    
    # fifo queue
    queue = [root_node]
    
    best_path = None
    best_cost = float('inf')

    while queue:
        current_node = queue.pop(0)
        if len(current_node.visited) == total_cities:
            return_cost = cost_matrix[current_node.current_city][start_city]
            if return_cost != float('inf'):
                total_route_cost = current_node.total_cost + return_cost
                if total_route_cost < best_cost:
                    best_cost = total_route_cost
                    best_path = current_node.path + [start_city]
            continue
            
        children = current_node.generate_children(cost_matrix, total_cities)
        queue.extend(children)

    return best_path, best_cost

if __name__ == "__main__":
    N = 6
    my_cities = generate_cities(N)
    my_matrix = generate_cost_matrix(my_cities, is_symmetric=True, density=1.0)
    best_route, lowest_cost = run_bfs(my_matrix, N, start_city=0)
    print(f"route: {best_route}, cost: {lowest_cost:.2f}")