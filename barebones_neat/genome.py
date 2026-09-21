from .connection_gene import ConnectionGene
import math
import random

class Genome:
    def __init__(self, input_size, output_size, chances, mutation_power):
        self.input_size = input_size
        self.output_size = output_size
        self.mutation_power = mutation_power
        self.chances = chances
        self.connections = {} # innovation num: ConnectionGene
        self.biases = {} # node_id: bias
        self.nodes = [] # tuples: (node_id, "in"/"out"/"hidden")
        next_num = 1
        for i in range(input_size):
            self.nodes.append((next_num, "in"))
            self.biases[next_num] = 0.0
            next_num += 1
        for i in range(output_size):
            self.nodes.append((next_num, "out"))
            self.biases[next_num] = 0.0
            next_num += 1
        next_innov = 1
        for i in range(input_size):
            for q in range(output_size):
                from_id = i + 1
                to_id = input_size + q + 1
                weight = random.uniform(-1, 1)
                self.connections[next_innov] = ConnectionGene(from_id, to_id, weight, next_innov)
                next_innov += 1

    def mutate(self, neat_class):
        choice = random.random() * 100
        if choice > 100 - self.chances["new_node"]:
            self.new_node(neat_class)
        elif choice > 100 - self.chances["new_node"] - self.chances["new_connection"]:
            self.new_connection(neat_class)
        elif choice > 100 - self.chances["new_node"] - self.chances["new_connection"] - self.chances["change_weight"]:
            self.change_weight()
        else:
            self.change_bias()

    def new_node(self, neat_class):
        if not self.connections:
            return
        active_conns = [c for c in self.connections.values() if c.enabled]
        if not active_conns:
            return
        split_gene = random.choice(active_conns)
        split_conn_tuple = (split_gene.from_node_id, split_gene.to_node_id)
        split_gene.enabled = False
        if split_conn_tuple not in neat_class.node_ids:
            neat_class.node_ids[split_conn_tuple] = neat_class.next_node_id
            node_id = neat_class.next_node_id
            neat_class.next_node_id += 1
        else:
            node_id = neat_class.node_ids[split_conn_tuple]
        self.nodes.append((node_id, "hidden"))
        self.biases[node_id] = 0.0
        conn1_tuple = (split_gene.from_node_id, node_id)
        if conn1_tuple not in neat_class.innovation_nums:
            neat_class.innovation_nums[conn1_tuple] = neat_class.next_innovation_num
            innov1 = neat_class.next_innovation_num
            neat_class.next_innovation_num += 1
        else:
            innov1 = neat_class.innovation_nums[conn1_tuple]
        self.connections[innov1] = ConnectionGene(split_gene.from_node_id, node_id, 1.0, innov1)
        conn2_tuple = (node_id, split_gene.to_node_id)
        if conn2_tuple not in neat_class.innovation_nums:
            neat_class.innovation_nums[conn2_tuple] = neat_class.next_innovation_num
            innov2 = neat_class.next_innovation_num
            neat_class.next_innovation_num += 1
        else:
            innov2 = neat_class.innovation_nums[conn2_tuple]
        self.connections[innov2] = ConnectionGene(node_id, split_gene.to_node_id, split_gene.weight, innov2)

    def new_connection(self, neat_class):
        attempts = 0
        while attempts < 100:
            node_a = random.choice(self.nodes)[0] # Pak enkel de ID-integer uit de tuple
            node_b = random.choice(self.nodes)[0]
            if self.check_forward(node_a, node_b) and not self.check_duplicate(node_a, node_b):
                conn_tuple = (node_a, node_b)
                if conn_tuple not in neat_class.innovation_nums:
                    neat_class.innovation_nums[conn_tuple] = neat_class.next_innovation_num
                    innov_num = neat_class.next_innovation_num
                    neat_class.next_innovation_num += 1
                else:
                    innov_num = neat_class.innovation_nums[conn_tuple]
                self.connections[innov_num] = ConnectionGene(node_a, node_b, random.uniform(-1, 1), innov_num)
                break
            attempts += 1

    def change_weight(self):
        if not self.connections:
            return
        chosen_gene = random.choice(list(self.connections.values()))
        chosen_gene.weight += random.gauss(0, self.mutation_power)

    def change_bias(self):
        if not self.biases:
            return
        chosen_node_id = random.choice(list(self.biases.keys()))
        self.biases[chosen_node_id] += random.gauss(0, self.mutation_power)

    def check_duplicate(self, node_a, node_b):
        for gene in self.connections.values():
            if gene.from_node_id == node_a and gene.to_node_id == node_b:
                return True
        return False

    def check_forward(self, node_a, node_b):
        type_a = None
        type_b = None
        for n_id, n_type in self.nodes:
            if n_id == node_a:
                type_a = n_type
            if n_id == node_b:
                type_b = n_type
        if type_a == "out" or type_b == "in" or node_a == node_b:
            return False
        return True