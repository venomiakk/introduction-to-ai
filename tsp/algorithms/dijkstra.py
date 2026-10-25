import heapq
from city import generate_cities, generate_cost_matrix
from tree_node import TreeNode

def run_dijkstra(cost_matrix, total_cities, start_city=0):
    root_node = TreeNode(
        current_city=start_city, 
        path=[start_city], 
        visited={start_city}, 
        total_cost=0.0
    )
    
    # priority queue (min-heap)
    pq = [(0.0, id(root_node), root_node)]
    
    best_path = None
    best_cost = float('inf')
    
    while pq:
        # getting cheapest node
        current_cost, _, current_node = heapq.heappop(pq)

        if current_cost >= best_cost:
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
                heapq.heappush(pq, (child.total_cost, id(child), child))
                
    return best_path, best_cost


if __name__ == "__main__":
    
    N = 6
    my_cities = generate_cities(N)
    my_matrix = generate_cost_matrix(my_cities, is_symmetric=True, density=1.0)
    path, cost = run_dijkstra(my_matrix, total_cities=N, start_city=0)
    print(f"route: {path}, cost: {cost:.2f}")