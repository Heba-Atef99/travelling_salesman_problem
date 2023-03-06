import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
global dist_mat 

class City:
    dist_mat = None

    def __init__(self, n, x, y) -> None:
        self.name = n
        self.x = x
        self.y = y

    @classmethod
    def set_dist_mat(cls, d):
        cls.dist_mat = d

    def get_distance(self, c2):
        # return np.linalg.norm([self.x - c2.x, self.y - c2.y], 2)
        return City.dist_mat[int(self.name)-1, int(c2.name)-1]
    
    def __repr__(self) -> str:
        return f'Name: {self.name} X: {self.x} Y: {self.y}\n'
    
    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name

class Chromosome:
    def __init__(self, g) -> None:
        self.genes = g
        self.cost = self.calc_cost()
        self.fitness = 1 / self.cost
    
    def calc_cost(self):
        cost = 0
        c1 = self.genes[0]
        for c2 in self.genes[1:]:
            cost += c1.get_distance(c2)
            c1 = c2
        return cost
    
    def update_fitness(self):
        self.cost = self.calc_cost()
        self.fitness = 1 / self.cost

    def __repr__(self) -> str:
        return f'Genes: {self.genes}\nCost: {self.cost}\nFitness: {self.fitness}'

class Population:
    def __init__(self, c) -> None:
        self.chromosomes = c
    
    def get_elite(self, elite_num):
        # sorte the chroms
        sorted_chroms = sorted(self.chromosomes, key= lambda c: c.fitness, reverse=True)
        return sorted_chroms[:elite_num]

    def __repr__(self) -> str:
        return f'Chromosomes: {self.chromosomes}'

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

def initialize_population(possible_chrom, pop_size):
    chroms = np.array([])
    for i in range(pop_size):
        c = Chromosome(np.random.permutation(possible_chrom))
        # return to the start city again
        c.genes = np.append(c.genes, c.genes[0])
        c.update_fitness()

        chroms = np.append(chroms, c)
    
    return Population(chroms)

# **********************************************************END OF ININTIALIZATION FUNCTIONS***************************************************************************
def tournament_selection(population, size):
    population_size = len(population.chromosomes)
    parents_idx = np.random.choice(population_size - 1, size= size, replace=False)
    chroms= population.chromosomes[parents_idx]
    chroms = sorted(chroms, key= lambda c: c.fitness, reverse=True)
    return [copy.deepcopy(chroms[0]), copy.deepcopy(chroms[1])]

def PMC(p1, p2):
    # get the cross point
    chrom_size = len(p1.genes)
    cross_points = np.random.choice(np.arange(1, chrom_size-1), size= 2, replace=False)
    # cross_points = np.random.choice(chrom_size, size= 2, replace=False)
    
    child = copy.deepcopy(p2)
    for i in range(min(cross_points), max(cross_points)+1):
        # get the index with the replacing value in the child
        replacing_index = np.where(child.genes == p1.genes[i])[0][0]

        # first replace the value that will be redundant in the child
        child.genes[replacing_index] = child.genes[i]

        # second replace the corresponding value of the parent in the child
        child.genes[i] = p1.genes[i]

    # To create a feasible solution
    child.genes[-1] = child.genes[0]
    return child

def crossover(parent1, parent2, prob):
    random_prop = np.random.rand()
    if random_prop < prob:
        child1 = PMC(parent1, parent2)   
        child2 = PMC(parent2, parent1)
        child1.update_fitness()
        child2.update_fitness()
        selected = sorted([child1, child2, parent1, parent2], key= lambda c: c.fitness, reverse=True)
        return selected[:2]
        # return [child1, child2]

    return [parent1, parent2]

def inversion_mutation(chrom):
    chrom_size = len(chrom.genes)
    cross_points = np.random.choice(np.arange(1, chrom_size-1), size= 2, replace=False)
    # cross_points = np.random.choice(chrom_size, size= 2, replace=False)
    start, stop = min(cross_points), max(cross_points)+1
    chrom.genes[start:stop] = np.flip(chrom.genes[start:stop])
    return chrom

def mutation(generation, elite_num, mutation_size, prob):
    gen_size = len(generation.chromosomes)
    # random_chroms = np.random.choice(np.arange(elite_num, gen_size), size= mutation_size, replace=False)
    random_chroms = np.random.choice(np.arange(elite_num, gen_size), size= int(gen_size * prob), replace=False)
    for c in random_chroms:
        # random_prop = np.random.rand()
        # if random_prop < prob:
        generation.chromosomes[c] = inversion_mutation(generation.chromosomes[c])
        generation.chromosomes[c].update_fitness()
    
    return generation

def genetic_algo(data, crossover_prob, population_size, mutation_prob, elite_number, max_loops):
    dist_mat = gen_distance_matrix(data)
    City.set_dist_mat(dist_mat)

    # 1. Initialize population
    cities = initialize_cities(data)
    old_gen = initialize_population(cities, population_size)
    selcetion_pop = int(0.5*population_size)

    for i in range(max_loops):
        # initialize the new generation with the elite
        elite = old_gen.get_elite(elite_number)

        # fill the new generation with the new gen
        # first fill with elite
        new_gen = Population(elite)

        # second fill with children or parents
        while len(new_gen.chromosomes) < population_size:
            # select parents from old generation
            parent1, parent2 = tournament_selection(old_gen, selcetion_pop)
            # parent2 = tournament_selection(old_gen, selcetion_pop)

            # Perform Crossover
            crossed = crossover(parent1, parent2, crossover_prob)

            # add the output from the crossover operation
            new_gen.chromosomes = np.append(new_gen.chromosomes, crossed)

        # perform mutation
        new_gen = mutation(new_gen, elite_number, 7, mutation_prob)
        # print(new_gen.get_elite(1))
        old_gen = copy.deepcopy(new_gen)
    
    best_sol = new_gen.get_elite(1)
    return best_sol

if __name__ == '__main__':
    df = pd.read_csv("15-Points.csv")
    # df = df.iloc[:3]
    dist_mat = gen_distance_matrix(df)
    crossover_prob = 0.7
    # population_size = 5
    population_size = 400
    mutation_prob = 0.3
    elatism_number = 100
    max_loops = 100

    sol = genetic_algo(df, crossover_prob, population_size, mutation_prob, elatism_number, max_loops)
    plot_sol(sol[0].genes, sol[0].cost, "Genetics Algo", df)
    # print(sol)