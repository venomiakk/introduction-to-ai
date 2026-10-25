from city import generate_cities, generate_cost_matrix

def run_nn(cost_matrix, total_cities, start_city=0):
    visited = {start_city}
    path = [start_city]
    current_city = start_city
    total_cost = 0.0
    
    # visiting cities
    while len(visited) < total_cities:
        nearest_city = None
        min_cost = float('inf')
        for next_city in range(total_cities):
            if next_city not in visited:
                cost = cost_matrix[current_city][next_city]
                if cost < min_cost:
                    min_cost = cost
                    nearest_city = next_city

        if nearest_city is None:
            return None, float('inf') 
            
        # move to next city
        visited.add(nearest_city)
        path.append(nearest_city)
        total_cost += min_cost
        current_city = nearest_city
        
    # back to start
    return_cost = cost_matrix[current_city][start_city]
    if return_cost == float('inf'):
        return None, float('inf')
        
    total_cost += return_cost
    path.append(start_city)
    
    return path, total_cost


if __name__ == "__main__":    
    N = 6
    my_cities = generate_cities(N)
    my_matrix = generate_cost_matrix(my_cities, is_symmetric=True, density=1.0)
    best_path, cost = run_nn(my_matrix, total_cities=N, start_city=0)
    print(f"route: {best_path}, cost: {cost:.2f}")