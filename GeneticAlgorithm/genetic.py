import unittest
import itertools as it
from random import randrange, randint, random
import numpy as np

# @incomplete - should probably use numpy arrays
# @incomplete - change the range of fitness to be a given fitness function range

class Genetic():
    """Genetic holds the functions necessary to perform a genetic algorithm on
    a given problem space and solution shape
    the user provides what the population members should look like as well as 
    the function that determines their fitness and the algorithm will optimize
    a solution to the problem using genetics.
    
    :member population: all of the potential solutions as a given point in runtime  
    :member population_size: a hyper-parameter that determines how many members should be in the population at one time  
    :member fitness: a function, provided by the user, that determines both how well a member solves the problem and if it is valid
    :member number_of_mates: a hyper-parameter that indicates how many members of the population should mate each generation
    :member mutation_rate: a hyper-parameter indicated how high of a chance a feature in the member should have to mutate
    :member iterations: a hyper-parameter that indicates how many generations of the genetic algorithm should run
    :member maximize: a hyper-parameter that determines if we want to minimize or maximize the fitness function
    :member feature_parameter: a hyper-parameter that tells the algorithm how to generate new solutions
    :member member_shape: indicates what a member of the population should look like
    :member number_of_parents_to_keep: a hyper-parameter indicating how many of the previous generation's best should remain in the new generation
    :member suppress_output: tells us to print or not
    
    :function __init__: initializes the hyper-parameters and the population
    :function run: the main loop for the genetic algorithm; for some number of iterations, finds the best members, breeds, crossover/mutates, and then refills
    :function crossover: takes two members and mixes them, returns two new children
    :function mutate: takes a member and has a chance to mutate features
    :function introduce_new_blood: generates new members of the population until the population size is reached
    :function create_new_member: creates a new member according to criteria
    :function pick_mate: based on the fitness of a member, select the best members to mate
    """
    def __init__(self, vector_length, population_size, fitness_function, check_member=None, feature_parameters=None, maximize=True, number_of_mates=10, mutation_rate=.05, number_of_iterations=100, keep_parents=2, suppress_output=False):
        # initializes the members for the algorithm
        # initializes the population according to the parameters
        # 
        # So we need to make a population of feature vectors that are the same length and whose features are
        # in the same range for each feature
        # For now, we can just assume that the features are all normalized on the interval [0, 1)
        # 
        # TODO: give the user a cull threshold that tells the algorithm to remove any members of the
        #       population that score below a certain threshold
        # TODO: write parameter documentation for this class
        self.population = []
        self.population_size = population_size
        self.fitness = fitness_function  # function that scores a member of the population  - note it could be that the user decides to have this function check the validity and give it a low score rather than remove the thing immediately
        self.number_of_mates = number_of_mates
        self.mutation_rate = mutation_rate  # chance of a feature getting mutated
        self.iterations = number_of_iterations
        self.maximize = maximize  # if we are maximizing the fitness function
        self.feature_parameters = feature_parameters
        self.vector_length = vector_length
        self.member_shape = None  # TODO: implement this
        self.number_of_parents_to_keep = keep_parents
        self.suppress_output = suppress_output
        
        for _ in range(population_size):
            new_member = self.create_new_member()
            self.population.append(new_member)
        
    def run(self):
        # after initializing the population, run new generations, evaluate,
        # and report back to user until we converge, or until we run the max number of generations
        # returns a tuple containing the best member of the population followed by the entire population
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
            # the way that the reporting is done will like change over time and will likley
            # not be nearly as verbose as this as this is a library function
            if i % 100 == 0:
                fitness = self.fitness(best)
                print(f'{best} = {fitness}')
        
        avg = sum([self.fitness(n) for n in self.population]) / len(self.population)
        print(f"\n\navg: {avg}\nbest:{best}")
        
        return best, self.population
    
    @staticmethod
    def crossover(a, b):
        # mixes two member of the population using random number generation
        # returns that new member
        # 
        # there are potentially other ways that we could 'crossover' solutions
        # that could be specified by the user. Another, for example, would be
        # choosing feature-by-feature whether to use parent a or b rather than choosing a cut
        # point and swapping.
        
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
    
    # def report(self, breeding_pool, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    #     # reports the most fit as well as the average fitness across all members
    #     # of the breeding pool
    #     # this will show an arrow representing how close to the max iterations we are
    #     # and it will update the values printed rather than print over and over
    #     avg = sum([self.fitness(n) for n in breeding_pool])
    #     best = self.fitness(breeding_pool[0])
    #     percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    #     filledLength = int(length * iteration // total)
    #     bar = fill * filledLength + '-' * (length - filledLength)
    #     print(f'\ravg: {avg}, best: {best}{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    #     # Print New Line on Complete
    #     if iteration == total: 
    #         print()
    #     return

# TODO: write unittests where applicable... might be hard given that a lot of this is random so 
#       expected output is hard to come by. but we can still check validity
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
    # 
    # NOTE: this particular function is used strictly to test the Genetic class - it will later be commented out and
    #       unusable to those that use the library
    w1, w2, w3, w4, w5 = vector
    return w1 + w2 - w3 - w4 + w5


if __name__ == "__main__":
    population_size = 30
    number_of_mates = 10
    iterations = 1000
    mutation_rate = .05
    keep_parents = 2
    ga = Genetic(5, population_size, fitness, number_of_mates=number_of_mates, number_of_iterations=iterations, mutation_rate=mutation_rate, keep_parents=keep_parents)
    ga.run()