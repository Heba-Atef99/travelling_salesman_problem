from nn import NN_algo, gen_distance_matrix
from genetics import genetic_algo
from aco import aco_algo

import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib

def plot_cities(cities_x, cities_y, names):
    plt.scatter(cities_x, cities_y)
    for n, x, y in zip(names, cities_x, cities_y):
        plt.annotate(n, (x, y), fontsize=12)

def plot_sol(sol, cost, algo, cities_df, is_list=False):
    plt.title("TSP Using " + algo + '\n Cost: ' + str(cost))
    plot_cities(cities_df['x'], cities_df['y'], cities_df['City'])
    for i in range(len(sol)-1):        
        if is_list:
            c1_x, c1_y = cities_df.iloc[sol[i]-1]['x'], cities_df.iloc[sol[i]-1]['y']
            c2_x, c2_y = cities_df.iloc[sol[i+1]-1]['x'], cities_df.iloc[sol[i+1]-1]['y']

        else:
            c1_x, c1_y = sol[i].x, sol[i].y
            c2_x, c2_y = sol[i+1].x, sol[i+1].y

        plt.plot([c1_x, c2_x], [c1_y, c2_y], color='red')
        plt.pause(1.15)

    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("15-Points.csv")
    dist_mat = gen_distance_matrix(df)

    # **************** NN ****************
    _, cost3, sol3 = NN_algo(df)
    
    # **************** GENETICS ****************
    crossover_prob = 0.7
    population_size = 400
    mutation_prob = 0.3
    elatism_number = 100
    max_loops = 100

    sol1 = genetic_algo(df, crossover_prob, population_size, mutation_prob, elatism_number, max_loops)
    print(sol1)

    # **************** ACO ****************
    population_size = 100
    max_loops = 10

    # α =3.0 to 5.0, β = 3.0, m = 50 to 800, ρ = 0.8
    rho = 0.7
    alpha = 2
    beta = 1
    sol2 = aco_algo(df, rho, alpha, beta, population_size, max_loops)


    plot_sol(sol3, cost3, "Nearest Neighbor Algo", df, True)
    plot_sol(sol1[0].genes, sol1[0].cost, "Genetics Algo", df)
    plot_sol(sol2[0].genes, sol2[0].cost, "Ant Colony Algo", df)
