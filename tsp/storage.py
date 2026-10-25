import pickle
import os
from city import generate_cities, generate_cost_matrix

def save_data(cities, cost_matrix, filename):
    filename = "data/" + filename + ".pkl"
    with open(filename, 'wb') as f:
        pickle.dump((cities, cost_matrix), f)

def load_data(filename):
    filename = "data/" + filename + ".pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: '{filename}'")
    
    with open(filename, 'rb') as f:
        cities, cost_matrix = pickle.load(f)
    return cities, cost_matrix


def generate_initial_data():
    for n in range(5,21):
        filename = f'tsp_data_{n}'
        cities = generate_cities(n)

        matrix = generate_cost_matrix(cities, True, 1.0)
        newfilename = filename + "_symmetric_1_0"
        save_data(cities, matrix, newfilename)

        matrix = generate_cost_matrix(cities, False, 1.0)
        newfilename = filename + "_asymmetric_1_0"
        save_data(cities, matrix, newfilename)

        matrix = generate_cost_matrix(cities, True, 0.8)
        newfilename = filename + "_symmetric_0_8"
        save_data(cities, matrix, newfilename)

        matrix = generate_cost_matrix(cities, False, 0.8)
        newfilename = filename + "_asymmetric_0_8"
        save_data(cities, matrix, newfilename)

if __name__ == "__main__":
    # generate_initial_data()
    cities, matrix = load_data("tsp_data_5_symmetric_1_0")
    print(cities)
    print(matrix)
    