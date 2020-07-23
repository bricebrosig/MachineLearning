import unittest
import itertools as it
from random import randrange, randint, random
import numpy as np

class Genetic():
    def __init__(self, vector_length, population_size, fitness_function, feature_parameters=None, maximize=True, number_of_mates=10, mutation_rate=.05, number_of_iterations=100, keep_parents=2):
        # initializes the members for the algorithm
        # initializes the population according to the parameters
        # if no fitness function is given then it will use the default
        # 
        # So we need to make a population of feature vectors that are the same length and whose features are
        # in the same range for each feature
        # For now, we can just assume that the features are all normalized on the interval [0, 1)
        self.population = []
        self.population_size = population_size
        self.fitness = fitness_function
        self.number_of_mates = number_of_mates
        self.mutation_rate = mutation_rate  # chance of a feature getting mutated
        self.iterations = number_of_iterations
        self.maximize = maximize  # if we are maximizing the fitness function
        self.feature_parameters = feature_parameters
        self.vector_length = vector_length
        self.number_of_parents_to_keep = keep_parents
        
        for _ in range(population_size):
            new_member = self.create_new_member()
            self.population.append(new_member)
        
    def run(self):
        # after initializing the population, run new generations, evaluate,
        # and report back to user until we converge, or until we run the max number of generations
        for i in range(self.iterations):
            next_generation = []
            
            population_scores = [self.fitness(n) for n in self.population]
            
            best = self.population[population_scores.index(max(population_scores))]
            
            # breed/crossover/mutate
            for _ in range(self.number_of_mates):
                mate_a, mate_b = self.population[self.pick_mate(population_scores)], self.population[self.pick_mate(population_scores)]
                child_a, child_b = self.crossover(mate_a, mate_b)
                
                self.mutate(child_a)  # doesn't necessarily mutate, just has the chance to mutate
                self.mutate(child_b)
                
                next_generation.append(child_a)
                next_generation.append(child_b)
                
            # keep the best
            next_generation += [best]
            for _ in range(self.number_of_parents_to_keep):
                next_generation += [self.population[self.pick_mate(population_scores)]]
            self.population = next_generation[:]

            # fill in the rest with randoms
            self.introduce_new_blood()
            
            # report
            if i % 100 == 0:
                fitness = self.fitness(best)
                print(f'{best} = {fitness}')
        
        avg = sum([self.fitness(n) for n in self.population]) / len(self.population)
        print(f"\n\navg: {avg}\nbest:{best}")
    
    @staticmethod
    def crossover(a, b):
        # mixes two member of the population using random number generation
        # returns that new member
        
        cut_point_a = randint(0, len(a))
        cut_point_b = randint(0, len(b))
        
        offspring_a = a[:cut_point_a] + b[cut_point_a:]
        offspring_b = b[:cut_point_b] + a[cut_point_b:]
        
        return offspring_a, offspring_b
    
    def mutate(self, a):
        # mutates a member of the population using random number generation
        # stores the mutated member in the passed parameter, a
        for element in a:
            if random() <= self.mutation_rate:
                element = random()  # re-roll that trait
        return
    
    def introduce_new_blood(self):
        # fills in the population with random members until full
        while len(self.population) < self.population_size:
            self.population.append(self.create_new_member())
        
    def create_new_member(self):
        # this function generates a new member of the population based off of the parameters provided
        # in the future this will have to check the shape of the member and the parameter list for that member
        return [random() for _ in range(self.vector_length)]
    
    def pick_mate(self, scores):
        # give the current population, choose which ones should breed based
        # off of their score on the fitness function
        # credit where due : https://github.com/gmichaelson/GA_in_python/blob/master/GA%20example.ipynb
        # pretty clever way to select mates based off of their probability
        # NOTE: we might need to change this in the future if the use decides to minimize their fitness function?
        array = np.array(scores)
        temp = array.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(array))

        fitness = [len(ranks) - x for x in ranks]
        cum_scores = fitness[:]
        
        for i in range(1,len(cum_scores)):
            cum_scores[i] = fitness[i] + cum_scores[i-1]
            
        probs = [x / cum_scores[-1] for x in cum_scores]
        
        rand = random()
        for i in range(0, len(probs)):
            if rand < probs[i]:
                return i
    
    def report(self, breeding_pool, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        # reports the most fit as well as the average fitness across all members
        # of the breeding pool
        # this will show an arrow representing how close to the max iterations we are
        # and it will update the values printed rather than print over and over
        avg = sum([self.fitness(n) for n in breeding_pool])
        best = self.fitness(breeding_pool[0])
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\ravg: {avg}, best: {best}{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
        return

class TestGenetic(unittest.TestCase):
    def test_initialize(self):
        # this test will ensure that the initialization of the genetic algorithms members work
        # properly with all the different parameters that one could pass to it.
        return True
    
    def test_crossover(self):
        # testing that the new member that was got from mixing two other members is, indeed, valid
        ga = Genetic(5)
        print(ga.crossover([1,2,3,4,5], [6,7,8,9,10]))
        return True
    
    def test_mutate(self):
        # testing that the mutated member is still valid
        return True
    
    def test_mutate_with_copy(self):
        # test that the mutated member is valid and that the passed member did 
        # not get mutated (in both meanings of the word)
        return 
    

def fitness(vector):
    # function that determines the fitness of a member. will be used to
    # determine if a member of the population is fit to continue breeding or not
    # returns a floating point value between 0 and 1
    # 
    # it will end up that the fitness function will not be defined by the class, rather by the user as 
    # we have no real way to intuit what they will want to do with their data structure that they pass
    # as a member to the population. 
    # So, for testing we can use what is below but in the future this will be completely user defined to 
    # increase the generality of the library
    w1, w2, w3, w4, w5 = vector
    return w1 + w2 - w3 - w4 + w5


if __name__ == "__main__":
    population_size = 30
    number_of_mates = 10
    ga = Genetic(5, population_size, fitness, number_of_mates=number_of_mates, number_of_iterations=1000)
    ga.run()