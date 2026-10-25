class TreeNode:
    __slots__ = ['current_city', 'path', 'visited', 'total_cost']

    def __init__(self, current_city, path, visited, total_cost):
        self.current_city = current_city  
        self.path = path 
        self.visited = visited
        self.total_cost = total_cost

    def generate_children(self, cost_matrix, total_cities):
        children = []
        
        for next_city in range(total_cities):
            if next_city not in self.visited:

                step_cost = cost_matrix[self.current_city][next_city]

                if step_cost != float('inf'):
                    
                    new_path = list(self.path)
                    new_path.append(next_city)
                    
                    new_visited = set(self.visited)
                    new_visited.add(next_city)
                    
                    new_cost = self.total_cost + step_cost

                    child_node = TreeNode(
                        current_city=next_city,
                        path=new_path,
                        visited=new_visited,
                        total_cost=new_cost
                    )
                    children.append(child_node)
                    
        return children

