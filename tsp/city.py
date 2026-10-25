import random
import math

class City:
    def __init__(self, city_id, x, y, z):
        self.id = city_id
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"City({self.id}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"

    def distance_to(self, other_city: City):
        return math.sqrt(
            (self.x - other_city.x) ** 2 +
            (self.y - other_city.y) ** 2 +
            (self.z - other_city.z) ** 2
        )

def generate_cities(n):
    cities = []
    for i in range(n):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        z = random.uniform(0, 50)
        cities.append(City(i, x, y, z))
    return cities


def generate_cost_matrix(cities, is_symmetric=True, density=1.0):
    n = len(cities)
    matrix = [[float('inf')] * n for _ in range(n)]
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges_to_keep_count = int(len(all_possible_edges) * density)
    kept_edges = set(random.sample(all_possible_edges, edges_to_keep_count))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0 
                continue
                
            edge_pair = (i, j) if i < j else (j, i)
            
            if edge_pair in kept_edges:
                dist = cities[i].distance_to(cities[j])
                
                if not is_symmetric:
                    z_diff = cities[j].z - cities[i].z
                    if z_diff > 0:
                        dist *= 1.1
                    elif z_diff < 0:
                        dist *= 0.9
                        
                matrix[i][j] = dist
                
    return matrix

if __name__ == "__main__":
    num_cities = 5
    cities = generate_cities(num_cities)
    cost_matrix = generate_cost_matrix(cities, is_symmetric=False, density=1.0)

    print("Cities:")
    for city in cities:
        print(f"{city}")

    print("\nCost Matrix:")
    for row in cost_matrix:
        print(row)