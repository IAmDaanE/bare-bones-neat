import random
import math
from .genome import Genome
from .compiled_genome import CompiledGenome

class NEAT:
    def __init__(self, population_size, input_size, output_size, crossover_chance, new_node_chance, new_connection_chance, change_weight_chance, change_bias_chance, mutation_power, similarity_treshold, excess_mult, disjoint_mult, weight_dif_mult, use_softmax):
        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.next_node_id = input_size + output_size + 1
        self.next_innovation_num = 1
        self.node_ids = {} 
        self.innovation_nums = {} 
        self.genome_pool = []
        self.specie_babies_amounts = []
        self.similarity_treshold = similarity_treshold
        self.excess_mult = excess_mult
        self.disjoint_mult = disjoint_mult
        self.weight_dif_mult = weight_dif_mult
        self.use_softmax = use_softmax
        self.chances = {
            "crossover": crossover_chance, 
            "clone": 100 - crossover_chance, 
            "new_node": new_node_chance, 
            "new_connection": new_connection_chance, 
            "change_weight": change_weight_chance, 
            "change_bias": change_bias_chance}
        for i in range(input_size):
            for q in range(output_size):
                from_id = i + 1
                to_id = input_size + q + 1
                self.innovation_nums[(from_id, to_id)] = self.next_innovation_num
                self.next_innovation_num += 1
        self.species = [[]]
        for i in range(population_size):
            genome_class = Genome(self.input_size, self.output_size, self.chances, mutation_power)
            init_conns = [(g.from_node_id, g.to_node_id) for g in genome_class.connections.values()]
            init_weights = {(g.from_node_id, g.to_node_id): g.weight for g in genome_class.connections.values()}
            self.species[0].append({
                "genome_class": genome_class, 
                "fitness_score": 0, 
                "compiled_genome_class": CompiledGenome(init_conns, init_weights, genome_class.biases.copy(), self.use_softmax, genome_class.nodes), 
            })
        self.representatives = []

    def compile_genomes(self):
        for specie in self.species:
            for dictionary in specie:
                gc = dictionary["genome_class"]
                current_conns = [(g.from_node_id, g.to_node_id) for g in gc.connections.values() if g.enabled]
                current_weights = {(g.from_node_id, g.to_node_id): g.weight for g in gc.connections.values() if g.enabled}
                dictionary["compiled_genome_class"] = CompiledGenome(current_conns, current_weights, gc.biases.copy(), self.use_softmax, gc.nodes)

    def assign_representatives(self):
        for specie in self.species:
            if specie:
                specie.sort(key=lambda x: x["fitness_score"])
                self.representatives.append(specie[-1])

    def merge_genomes(self, better_genome, worse_genome):
        child = Genome(self.input_size, self.output_size, better_genome.chances, better_genome.mutation_power)
        child.connections.clear()
        child.nodes = list(better_genome.nodes)
        equal_fitness = False
        all_innovations = sorted(set(better_genome.connections.keys()) | set(worse_genome.connections.keys()))
        for innov in all_innovations:
            in_better = innov in better_genome.connections
            in_worse = innov in worse_genome.connections
            if in_better and in_worse:
                if random.random() < 0.5:
                    child.connections[innov] = better_genome.connections[innov].clone()
                else:
                    child.connections[innov] = worse_genome.connections[innov].clone()
                    
                if not better_genome.connections[innov].enabled or not worse_genome.connections[innov].enabled:
                    if random.random() < 0.75:
                        child.connections[innov].enabled = False
            else:
                if equal_fitness:
                    if in_better and random.random() < 0.5:
                        child.connections[innov] = better_genome.connections[innov].clone()
                    elif in_worse and random.random() < 0.5:
                        child.connections[innov] = worse_genome.connections[innov].clone()
                else:
                    if in_better:
                        child.connections[innov] = better_genome.connections[innov].clone()  
        child.biases = better_genome.biases.copy()
        return child

    def sort_into_species(self):
        self.species = [[] for _ in range(len(self.representatives))]
        for genome in self.genome_pool:
            found_species = False
            i = 0
            while i < len(self.representatives):
                rep_dict = self.representatives[i]
                rep_genome = rep_dict["genome_class"]
                if self.calculate_distance(genome, rep_genome) <= self.similarity_treshold:
                    self.species[i].append({
                        "genome_class": genome, 
                        "fitness_score": 0, 
                        "compiled_genome_class": 0, 
                    })
                    found_species = True
                    break
                i += 1
            if not found_species:
                new_member_dict = {
                    "genome_class": genome, 
                    "fitness_score": 0, 
                    "compiled_genome_class": 0, 
                }
                self.species.append([new_member_dict])
                self.representatives.append(new_member_dict)
        for specie in self.species[:]:
            if len(specie) == 0:
                self.species.remove(specie)

    def create_new_generation(self):
        self.specie_babies_amounts = self.get_specie_babies_amounts()
        for q, specie in enumerate(self.species):
            for i in range(self.specie_babies_amounts[q]):
                self.produce_baby(q)

    def calculate_distance(self, genome_1, genome_2):
        g1size = len(genome_1.connections)
        g2size = len(genome_2.connections)
        N = max(g1size, g2size)
        if N < 20:
            N = 1
        innov1_set = set(genome_1.connections.keys())
        innov2_set = set(genome_2.connections.keys())
        max_innov1 = max(innov1_set) if innov1_set else 0
        max_innov2 = max(innov2_set) if innov2_set else 0
        limit_innov = min(max_innov1, max_innov2)
        all_innov_mismatches = innov1_set ^ innov2_set
        excess = 0
        disjoint = 0
        for innov in all_innov_mismatches:
            if innov > limit_innov:
                excess += 1
            else:
                disjoint += 1 
        matching_innovs = innov1_set & innov2_set
        weight_diff_sum = 0.0
        for innov in matching_innovs:
            weight_diff_sum += abs(genome_1.connections[innov].weight - genome_2.connections[innov].weight)
        avg_weight_diff = (weight_diff_sum / len(matching_innovs)) if matching_innovs else 0.0
        distance = (self.excess_mult * excess / N) + (self.disjoint_mult * disjoint / N) + (self.weight_dif_mult * avg_weight_diff)
        return distance

    def get_specie_babies_amounts(self):
        if not self.species:
            return []
        total_fitness_for_species = []
        for specie in self.species:
            specie_fitness_sum = 0
            for dictionary in specie:
                specie_fitness_sum += dictionary["fitness_score"]
            avg_fitness = specie_fitness_sum / len(specie) if len(specie) > 0 else 0
            total_fitness_for_species.append(avg_fitness)
        total_fitness = sum(total_fitness_for_species)
        if total_fitness == 0:
            base_amount = self.population_size // len(self.species)
            returned_values = [base_amount] * len(self.species)
            for i in range(self.population_size % len(self.species)):
                returned_values[i] += 1
            return returned_values
        ratio = self.population_size / total_fitness
        returned_values = []
        for avg_score in total_fitness_for_species:
            exact_babies = avg_score * ratio
            rounded_babies = max(1, round(exact_babies))
            returned_values.append(rounded_babies)
        check_addup = sum(returned_values)
        difference = self.population_size - check_addup
        while difference != 0:
            if difference > 0:
                chosen = random.randint(0, len(returned_values) - 1)
                returned_values[chosen] += 1
                difference -= 1
            else:
                valid_indices = [i for i, val in enumerate(returned_values) if val > 1]
                if not valid_indices:
                    break
                chosen = random.choice(valid_indices)
                returned_values[chosen] -= 1
                difference += 1
        return returned_values

    def produce_baby(self, specie_index): # either makes a clone or crossover for a specific specie
        sp_size = len(self.species[specie_index])
        self.species[specie_index].sort(key=lambda x: x["fitness_score"])
        if sp_size > 1:
            deleted_amount = math.floor(sp_size / 2)
            del self.species[specie_index][:deleted_amount]
            sp_size = len(self.species[specie_index])
        choice = random.random() * 100
        tournie_pool = []
        baby_genome = 0
        baby_genome_class = 0
        if choice < self.chances["crossover"] and sp_size > 1: # crossover
            if sp_size >= 4:
                while len(tournie_pool) < 4:
                    chosen = random.choice(self.species[specie_index])
                    if chosen not in tournie_pool:
                        tournie_pool.append(chosen)
                tournie_pool.sort(key=lambda x: x["fitness_score"])
                del tournie_pool[:2]
            else: 
                tournie_pool = [self.species[specie_index][0], self.species[specie_index][1]]
            genome_worse = tournie_pool[0]["genome_class"]
            genome_better = tournie_pool[1]["genome_class"]            
            baby_genome_class = self.merge_genomes(genome_better, genome_worse)            
            baby_genome_class.mutate(self)
        else: # clone   
            if sp_size >= 3:
                while len(tournie_pool) < 3:
                    chosen = random.choice(self.species[specie_index])
                    if chosen not in tournie_pool:
                        tournie_pool.append(chosen)
                tournie_pool.sort(key=lambda x: x["fitness_score"])
                baby_genome = tournie_pool[-1]
            else:
                baby_genome = self.species[specie_index][-1]
            baby_genome_class = baby_genome["genome_class"]
            baby_genome_class.mutate(self)     
        self.genome_pool.append(baby_genome_class)