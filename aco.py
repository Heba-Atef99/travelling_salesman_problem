import numpy as np
import pandas as pd
import copy
global dist_mat 

class City:
    dist_mat = None

    def __init__(self, n, x, y) -> None:
        self.name = n
        self.x = x
        self.y = y
        self.prob = 0

    @classmethod
    def set_dist_mat(cls, d):
        cls.dist_mat = d

    def get_distance(self, c2):
        return City.dist_mat[int(self.name)-1, int(c2.name)-1]
    
    def __repr__(self) -> str:
        return f'Name: {self.name} X: {self.x} Y: {self.y}\n'
    
    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name

class Agent:
    def __init__(self, p=[]) -> None:
        self.path = p
        if len(p) > 0:
            self.cost = self.calc_cost()
            self.fitness = 1 / self.cost
        else:
            self.cost = 0
            self.fitness = np.inf
    
    def calc_cost(self):
        cost = 0
        c1 = self.path[0]
        for c2 in self.path[1:]:
            cost += c1.get_distance(c2)
            c1 = c2
        return cost
    
    def update_fitness(self):
        self.cost = self.calc_cost()
        self.fitness = 1 / self.cost

    def __repr__(self) -> str:
        return f'Path: {self.path}\nCost: {self.cost}\nFitness: {self.fitness}'

class Colony:
    def __init__(self, a=[]) -> None:
        self.agents = a
    
    def get_elite(self, elite_num):
        # sorte the agents
        sorted_agents = sorted(self.agents, key= lambda c: c.fitness, reverse=True)
        return sorted_agents[:elite_num]

    def __repr__(self) -> str:
        return f'Agents: {self.agents}'

# ******************************************************** END OF CLASSES *****************************************************************************
def gen_distance_matrix(data):
    m = len(data)
    dist_mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            x1, y1 = data.iloc[i, 0:2]
            x2, y2 = data.iloc[j, 0:2]
            dist_mat[i, j] = np.linalg.norm([x1-x2, y1-y2], 2)

    return dist_mat

def initialize_cities(data):
    cities = np.array([])
    for i in range(len(data)):
        name, x, y = data.iloc[i][['City', 'x', 'y']]
        c = City(int(name), x, y)
        cities = np.append(cities, c)

    return cities

def initialize_colony(possible_agent, colony_size):
    agents = np.array([])
    for i in range(colony_size):
        c = Agent(np.random.permutation(possible_agent))
        # return to the start city again
        c.path = np.append(c.path, c.path[0])
        c.update_fitness()

        agents = np.append(agents, c)
    
    return Colony(agents)

# **********************************************************END OF ININTIALIZATION FUNCTIONS***************************************************************************
def update_phermon_mat(ph_mat, rho, colony):
    agents = colony.agents
    new_ph = copy.deepcopy(ph_mat) * (1-rho)

    for c in agents:
        sol = c.path
        ph_val = 1 / c.cost
        for i in range(len(sol)-1):
            c1 = sol[i].name - 1
            c2 = sol[i+1].name - 1

            new_ph[c1, c2] += ph_val        
            new_ph[c2, c1] += ph_val        

    return new_ph

def calc_prob_den(ph_mat, current_city, unvisited, alpha, beta):
    id = current_city.name - 1 
    term1 =  ph_mat[id, unvisited] ** alpha
    term2 = City.dist_mat[id, unvisited] ** -beta
    return np.dot(term1, term2)

def calc_prob(ph_mat, current_city, other_city, den, alpha, beta):
    id = current_city.name - 1 
    num = ph_mat[id, other_city.name-1]**alpha * current_city.get_distance(other_city)**-beta
    return num/den

def calc_cdf(cities):
    sum = 0
    cities = sorted(cities, key=lambda c: c.prob)
    for c in cities:
        sum += c.prob
        c.prob = sum
    return cities 

def choose_city(cities):
    random_prop = np.random.rand()
    for c in cities:
        if random_prop < c.prob:
            return c
        
def construct_solutions(cities, ph_mat, colony_size, alpha, beta):
    new_gen = Colony()

    # ** fill the generation with solutions **
    for i in range(colony_size):
        unvisited_mask = np.ones(len(cities), dtype=bool)
        unvisited_cities = copy.deepcopy(cities)

        #  ** construct solution ** 
        sol = Agent()

        # choose a random start city
        start_city = np.random.choice(cities)
        current_city = copy.deepcopy(start_city)

        while len(unvisited_cities) > 0:
            sol.path = np.append(sol.path, current_city)

            unvisited_mask[current_city.name-1] = False

            unvisited_cities = cities[unvisited_mask]
            if len(unvisited_cities) == 1:
                sol.path = np.append(sol.path, unvisited_cities[0])
                break

            # Get Propability
            den = calc_prob_den(ph_mat, current_city, unvisited_mask, alpha, beta)
            for c in unvisited_cities:
                c.prob = calc_prob(ph_mat, current_city, c, den, alpha, beta)
            
            unvisited_cities = calc_cdf(unvisited_cities)
            current_city = choose_city(unvisited_cities)

        # make the solution feasable
        sol.path = np.append(sol.path, start_city)
        sol.update_fitness()
        
        # add the solution to colony
        new_gen.agents = np.append(new_gen.agents, sol) 

    return new_gen

def aco_algo(data, rho, alpha, beta, colony_size, max_loops):
    dist_mat = gen_distance_matrix(data)
    City.set_dist_mat(dist_mat)

    # 1. Initialize colony
    cities = initialize_cities(data)
    generation = initialize_colony(cities, colony_size)
    
    # 2. Update phermones matrix
    cities_num = len(data['City'].unique())
    ph_mat = np.zeros((cities_num, cities_num))
    ph_mat = update_phermon_mat(ph_mat, rho, generation)

    for i in range(max_loops):
        # construct solution
        generation = construct_solutions(cities, ph_mat, colony_size, alpha, beta)
        # Update phermones matrix
        ph_mat = update_phermon_mat(ph_mat, rho, generation)

    best_sol = generation.get_elite(1)
    return best_sol

if __name__ == '__main__':
    df = pd.read_csv("15-Points.csv")
    dist_mat = gen_distance_matrix(df)
    colony_size = 100
    max_loops = 10
    # colony_size = 3
    # max_loops = 50

    # α =3.0 to 5.0, β = 3.0, m = 50 to 800, ρ = 0.8
    rho = 0.7
    alpha = 2
    beta = 1
    sol = aco_algo(df, rho, alpha, beta, colony_size, max_loops)
    print(sol)