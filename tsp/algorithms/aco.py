import random
from city import generate_cities, generate_cost_matrix

def run_aco(cost_matrix, total_cities, start_city=0, 
            num_ants=15, num_iterations=50, 
            alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100.0):

    # num_ants: number of ants to simulate per iteration
    # num_iterations: number of iterations to run the algorithm
    # alpha: importance of pheromone (higher = more influence)
    # beta: importance of heuristic (higher = more influence)
    # evaporation_rate: rate at which pheromone evaporates (0 < evaporation_rate < 1)
    # Q: constant used to determine how much pheromone to deposit (higher = more pheromone)

    pheromones = [[1.0 for _ in range(total_cities)] for _ in range(total_cities)]
    
    best_path = None
    best_cost = float('inf')
    
    for iteration in range(num_iterations):
        all_paths = []
        all_costs = []
        
        for ant in range(num_ants):
            current_city = start_city
            visited = {current_city}
            path = [current_city]
            tour_cost = 0.0
            
            valid_tour = True
            
            # ant routing through cities
            while len(visited) < total_cities:
                probabilities = []
                sum_prob = 0.0
                
                # calculate probabilities for moving to the next city
                for next_city in range(total_cities):
                    if next_city not in visited:
                        dist = cost_matrix[current_city][next_city]
                        if dist != float('inf'):
                            # tau = pheromone, eta = heuristic (1/distance)
                            tau = pheromones[current_city][next_city] ** alpha
                            eta = (1.0 / dist) ** beta
                            prob = tau * eta
                            
                            probabilities.append((next_city, prob))
                            sum_prob += prob
                        else:
                            probabilities.append((next_city, 0.0))
                
                if sum_prob == 0:
                    valid_tour = False
                    break
                    
                # selection of next city
                rand = random.uniform(0, sum_prob)
                cumulative = 0.0
                selected_city = None
                for city, prob in probabilities:
                    if prob > 0:
                        cumulative += prob
                        if cumulative >= rand:
                            selected_city = city
                            break
                            
                if selected_city is None:
                    selected_city = [c for c, p in probabilities if p > 0][-1]
                    
                # ant moves to the selected city
                dist = cost_matrix[current_city][selected_city]
                tour_cost += dist
                current_city = selected_city
                visited.add(current_city)
                path.append(current_city)
                
            # back to start
            if valid_tour:
                return_dist = cost_matrix[current_city][start_city]
                if return_dist != float('inf'):
                    tour_cost += return_dist
                    path.append(start_city)
                    
                    all_paths.append(path)
                    all_costs.append(tour_cost)

                    if tour_cost < best_cost:
                        best_cost = tour_cost
                        best_path = path

        # update pheromones
        for i in range(total_cities):
            for j in range(total_cities):
                pheromones[i][j] *= (1.0 - evaporation_rate)
                
        for i in range(len(all_paths)):
            path = all_paths[i]
            cost = all_costs[i]
            pheromone_to_deposit = Q / cost
            
            for j in range(len(path) - 1):
                from_city = path[j]
                to_city = path[j+1]
                pheromones[from_city][to_city] += pheromone_to_deposit
                
    return best_path, best_cost

if __name__ == "__main__":
    N = 10
    my_cities = generate_cities(N)
    my_matrix = generate_cost_matrix(my_cities, is_symmetric=True, density=1.0)
    path, cost = run_aco(my_matrix, total_cities=N, start_city=0, num_ants=20, num_iterations=100)
    print(f"route: {path}, cost: {cost:.2f}")