import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import sys
import os
import random
import shutil
import time
import math
import random

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):


        # Weights are how often an individual tile is going to be changed.
        EMPTY_WEIGHT = 0.09
        WALL_WEIGHT = .005
        COIN_BLOCK_WEIGHT = 0.005
        MUSH_BLOCK_WEIGHT = 0.04
        BREAKABLE_WEIGHT = 0.001
        COIN_WEIGHT = 0.01
        PIPE_WEIGHT = 0.005
        ENEMY_WEIGHT = 0.0002   

        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                match genome[y][x]:
                    case "-":
                        if random.random() < EMPTY_WEIGHT: 
                            self.mutate_do_something(genome, x,y)               
                    case "X":
                        if random.random() < WALL_WEIGHT:
                            self.mutate_do_something(genome, x,y)
                    case "?":
                        if random.random() < COIN_BLOCK_WEIGHT:
                            self.mutate_do_something(genome, x,y)
                    case "M":
                        if random.random() < MUSH_BLOCK_WEIGHT:
                            self.mutate_do_something(genome, x,y)                
                    case "B":
                        if random.random() < BREAKABLE_WEIGHT:
                            self.mutate_do_something(genome, x,y)               
                    case "o":
                        if random.random() < COIN_WEIGHT:
                            self.mutate_do_something(genome, x,y)
                    case "|":
                        if random.random() < PIPE_WEIGHT:
                            self.mutate_do_something(genome, x,y)
                        self.mutate_correct_pipe_section(genome,x,y)                        
                    case "T":
                        if random.random() < PIPE_WEIGHT:
                            self.mutate_do_something(genome, x,y)
                        self.mutate_correct_pipe_section(genome, x,y)        
                    case  "E": 
                        if random.random() < ENEMY_WEIGHT:
                            self.mutate_do_something(genome, x,y)

        return genome

    def mutate_do_something(self, genome, x,y):
        RANDOM_SWAP_WEIGHT = 0.15
        SWAP_TWO_WEIGHT = 0.05
        BECOME_RANDOM_WEIGHT = 0.02
        BECOME_AIR_WEIGHT = 0.07
        AREA_WEIGHT = .2

        if random.random() < RANDOM_SWAP_WEIGHT:
            self.mutate_swap_with_random(genome, x, y)
            return
        
        if random.random() < SWAP_TWO_WEIGHT:
            self.mutate_swap_two(genome, x, y)
            return

        if random.random() < BECOME_RANDOM_WEIGHT:
            self.mutate_becomes_random(genome, x, y)
            return

        if random.random() < BECOME_AIR_WEIGHT:
            self.mutate_becomes_air(genome, x, y)
            return
        
        # if random.random() < AREA_WEIGHT:
        #     self.mutate_area_becomes_other_area(genome, x, y)
        #     return

    def mutate_swap_with_random(self, genome, x,y):
        swap = genome[y][x]
        rand_x = random.randrange(1, width -1)
        rand_y = random.randrange(0, height-1)

        entry_two = genome[rand_y][rand_x]
        
        # Finish swap
        genome[y][x] = entry_two
        genome[rand_y][rand_x] = swap

    def mutate_swap_two(self, genome, x1, y1):

        x2 = random.randrange(1, width-1)
        y2 = random.randrange(0, height-1)

        swap = genome[y1][x1]
        entry_two = genome[y2][x2]
        
        # Finish swap
        genome[y1][x1] = entry_two
        genome[y2][x2] = swap   

    def mutate_becomes_random(self, genome, x,y):
        if (y <= 15):
            genome[y][x] = options[random.randint(0, len(options) - 1)]
        
    def mutate_becomes_air(self, genome, x,y):
        genome[y][x] = options[0]
        
    def mutate_area_becomes_air(self, genome, x,y):
        area = random.randrange(1,30)
        
        # TODO: Possibly change this to some kind of thing that cuts the area off instead of just giving up
        if (x - area < 1) or (x + area > width) \
        or (y - area < 1) or (y + area > height):
            return

        for a in range(x, x + area):
            for b in range(y, y + area):
                genome[b][a] = options[0]


    def mutate_area_becomes_other_area(self, genome, x,y):
        area = random.randrange(1,20)

        x2 = random.randrange(1, width-1 - area)
        y2 = random.randrange(0, height - area)
        
        # TODO: Possibly change this to some kind of thing that cuts the area off instead of just giving up
        if (x2 - area < 1) or (x2 + area > width) \
        or (y2 - area < 1) or (y2 + area > height):
            return

        if (x - area < 1) or (x + area > width) \
        or (y - area < 1) or (y + area > height):
            return

        for x in range(x2, x2 + area):
            for y in range(y2, y2 + area):
                swap = genome[y][x]
                entry_two = genome[y2][x2]
                
                # Finish swap
                genome[y][x] = entry_two
                genome[y2][x2] = swap  
                
        
    def mutate_correct_pipe_top(self, genome, x,y):
        if genome[y][x] == "T":
            if y >= 15:
                genome[y][x] = "-"
                return
            for a in range (y, 0):
                genome[a][x] = "|"


    def mutate_correct_pipe_section(self, genome, x,y):
        top = ""
        if genome[y][x] == "|":
            if y >= 15:
                genome[y][x] = "-"
                return
            
            for a in range (y, height):
                if genome[a][x] != "|":
                    genome[a][x] = "T"
                    top = a
                if a == height: return

            for a in range (y, 0):
                genome[a][x] = "|" 
                    
    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                # pass
                # STUDENT Which one should you take?  Self, or other?  Why?
                # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
                if other.genome[y][x] == "-" and random.random() < 0.5:
                        new_genome[y][x] = other.genome[y][x]
                else:
                # first heuristic checks the sky. 
                    if y < int(height * 0.80):
                        # checks enemy position:
                        if new_genome[y][x] == "E" and (new_genome[y+1][x] not in ["X", "?", "M", "B"]) and other.genome[y][x] != "E":
                            new_genome[y][x] == other.genome[y][x]

                        # prevents blocks from stacking on each other
                        if (new_genome[y][x] in ["X", "?", "M", "B"]) and (new_genome[y+1][x] in ["X", "?", "M", "B"]) and (other.genome[y][x] not in ["X", "?", "M", "B"]):
                            new_genome[y][x] = other.genome[y][x]

                        # checks whether there is a pipe segment or top
                        if (new_genome[y][x] in ["T", "|"]) and (other.genome[y][x] not in ["T", "|"]):
                            new_genome[y][x] = other.genome[y][x]

                        # checks if there's a wall at the sky
                        if new_genome[y][x] == "X" and other.genome[y][x] != "X":
                            new_genome[y][x] = other.genome[y][x]
                    else:
                        # check all ground related actions
                        # checks enemy position:
                        # checks if enemy is on a platorm
                        if y < height-1:
                            if new_genome[y][x] == "E" and new_genome[y+1][x] not in ["X", "?", "M", "B"] and other.genome[y][x] != "E":
                                    new_genome[y][x] = other.genome[y][x]

                        # Update based on stacked blocks
                        if y < (height - 2):
                            if new_genome[y][x] in ["X", "?", "M", "B"] and new_genome[y+1][x] in ["X", "?", "M", "B"] and other.genome[y][x] == "-":
                                if random.random() < 0.7:
                                    new_genome[y][x] = other.genome[y][x]

                        # ensures pipes are based properly
                        if y < height-1:
                            if (new_genome[y][x] in ["T", "|"]) and (new_genome[y + 1][x] not in ["|", "X"]) and other.genome[y+1][x] in ["X", "|"]:
                                new_genome[y+1][x] = other.genome[y+1][x]

                        # Update surroudnings in beginning
                        if x < int(right * 0.05) and y < height - 1:
                            if other.genome[y][x] == "-":
                                new_genome[y][x] = other.genome[y][x]

        # do mutation; note we're returning a one-element tuple here
        # for h in new_genome:
            # for w in h:
                # print(w, end="")
            # print()
        # exit()
        # call mutate
        new_genome = self.mutate(new_genome)

        return (Individual_Grid(new_genome),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.7, #increase weight
            negativeSpace=0.5, #decrease weight
            pathPercentage=0.6,
            emptyPercentage=0.6,
            linearity=-0.4, #adjust weight
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 3 # increase penalty
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = list(de)  # Convert to list for mutable modification
            x = de[0]
            de_type = de[1]
            choice = random.random()
            
            mutation_strategies = {
                "4_block": lambda: (x, de_type, offset_by_upto(de[2], height / 2, min=0, max=height - 1), not de[3] if choice >= 0.40 else de[3]),
                "5_qblock": lambda: (offset_by_upto(x, width / 8, min=1, max=width - 2), de_type, offset_by_upto(de[2], height / 2, min=0, max=height - 1), not de[3] if choice >= 0.40 else de[3]),
                "3_coin": lambda: (offset_by_upto(x, width / 8, min=1, max=width - 2), de_type, offset_by_upto(de[2], height / 2, min=0, max=height - 1)),
                "7_pipe": lambda: (offset_by_upto(x, width / 8, min=1, max=width - 2), de_type, offset_by_upto(de[2], 2, min=2, max=height - 4)),
                "0_hole": lambda: (offset_by_upto(x, width / 8, min=1, max=width - 2), de_type, offset_by_upto(de[2], 4, min=1, max=width - 2)),
                "6_stairs": lambda: (offset_by_upto(x, width / 8, min=1, max=width - 2), de_type, offset_by_upto(de[2], 8, min=1, max=height - 4), -de[3] if choice >= 0.40 else de[3]),
                "1_platform": lambda: (
                    offset_by_upto(x, width / 8, min=1, max=width - 2) if choice < 0.25 else de[0],
                    de_type,
                    offset_by_upto(de[2], 8, min=1, max=width - 2) if choice < 0.5 else de[2],
                    offset_by_upto(de[3], height, min=0, max=height - 1) if choice < 0.75 else de[3],
                    random.choice(["?", "X", "B"]) if choice >= 0.75 else de[4]
                ),
                "2_enemy": lambda: de  # No mutation for enemy type
            }
            
            # Apply mutation based on the design element type
            mutation_func = mutation_strategies.get(de_type, lambda: de)
            new_de = mutation_func()
            
            new_genome[to_change] = tuple(new_de)
            
        return new_genome


    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_DE


def generate_successors(population):
    results = []
    
    # Roulette selection
    roulette_cand = []
    fitness_sum = sum(p._fitness for p in population)
    cumulative_probabilities = []
    cumulative_prob = 0
    for p in population:
        cumulative_prob += p._fitness / fitness_sum
        cumulative_probabilities.append(cumulative_prob)
    
    random.shuffle(population)
    n = int(len(population))
    for _ in range(n):
        rand_val = random.random()
        selected = None
        for i, cum_prob in enumerate(cumulative_probabilities):
            if rand_val <= cum_prob:
                selected = population[i]
                break
        
        if selected is not None:
            roulette_cand.append(selected)
    
    # Tournament selection
    tournament_cand = []
    tournament_size = 2
    n = int(len(roulette_cand))
    for _ in range(n):
        tournament = random.sample(roulette_cand, tournament_size)
        winner = max(tournament, key=lambda x: x._fitness)
        tournament_cand.append(winner)
    
    # Remove individuals with genome length of 0
    tournament_cand = [ind for ind in tournament_cand if len(ind.genome) > 0]

    # Generate children from selected individuals
    for i in range(len(tournament_cand)):
        parent1 = tournament_cand[i]
        parent2 = random.choice(tournament_cand)
        results.extend(parent1.generate_children(parent2))

    return results

def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                    else Individual.empty_individual()
                    for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                            population,
                            batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        while True:
            try:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                        next_population,
                                        batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting GA process...")
                pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
