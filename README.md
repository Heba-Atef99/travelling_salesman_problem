# Travelling Salesman Problem

<div align="center">
  <img src="https://user-images.githubusercontent.com/54477107/223072867-efd659db-5d13-4873-814b-04392d308486.png">  
</div>

## Problem Definition
The travelling salesman problem (also called the travelling salesperson problem or TSP) asks the following question: "Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?"

The TSP has several applications even in its purest formulation, such as planning, logistics, and the manufacture of microchips.

# Solution Techniques
I solved this problem using three techniques which are Nearest Neighbor algorithm , Genetics algorithm, and Ant Colony algorithm.
I also used 15-points dataset which includes 15 cities the salesman shall visit.

## Nearest Neighbor Algorithm
The nearest neighbour algorithm is easy to implement and executes quickly, but it can sometimes miss shorter routes which are easily noticed with human insight, due to its "greedy" nature.

### Steps
1. Initialize all vertices as unvisited.
2. Select an arbitrary vertex, set it as the current vertex u. Mark u as visited.
3. Find out the shortest edge connecting the current vertex u and an unvisited vertex v.
4. Set v as the current vertex u. Mark v as visited.
5. If all the vertices in the domain are visited, then terminate. Else, go to step 3.

## Genetics Algorithm
<div align="center">
  <img src="https://user-images.githubusercontent.com/54477107/223075209-8d667dd3-b66c-4a3d-b683-84109b2a13c0.png">  
</div>

Genetic algorithms are metaheuristic search algorithms inspired by the process that supports the evolution of life. The algorithm is designed to replicate the natural selection process to carry generation, i.e. survival of the fittest of beings.

### Steps
1. Creating initial population.
2. Calculating fitness.
3. Selecting the best genes.
4. Crossing over.
5. Mutating to introduce variations.

## Ant Colony Algorithm
<div align="center">
  <img src="https://user-images.githubusercontent.com/54477107/223076439-8316c87e-464f-455c-a144-b2696520119d.jpg">  
</div>

Ant Colony algorithm is a metaheuristic search algorithm purely inspired from the foraging behaviour of ant colonies. Ants are eusocial insects that prefer community survival and sustaining rather than as individual species. They communicate with each other using sound, touch and pheromone. Pheromones are organic chemical compounds secreted by the ants that trigger a social response in members of same species. These are chemicals capable of acting like hormones outside the body of the secreting individual, to impact the behaviour of the receiving individuals. Since most ants live on the ground, they use the soil surface to leave pheromone trails that may be followed (smelled) by other ants.

Ants live in community nests and the underlying principle of ACO is to observe the movement of the ants from their nests in order to search for food in the shortest possible path.

### Steps
1. All ants are in their nest. There is no pheromone content in the environment. (For algorithmic design, residual pheromone amount can be considered without interfering with the probability)

2. Ants begin their search with equal (0.5 each) probability along each path. Clearly, the curved path is the longer and hence the time taken by ants to reach food source is greater than the other.

3. The ants through the shorter path reaches food source earlier. Now, evidently they face with a similar selection dilemma, but this time due to pheromone trail along the shorter path already available, probability of selection is higher.

4. More ants return via the shorter path and subsequently the pheromone concentrations also increase. Moreover, due to evaporation, the pheromone concentration in the longer path reduces, decreasing the probability of selection of this path in further stages. Therefore, the whole colony gradually uses the shorter path in higher probabilities. So, path optimization is attained.

# Animated Solution For Each Technique

https://user-images.githubusercontent.com/54477107/223085361-5dfd01af-001d-4850-ace4-a95556fa8fd1.mp4


