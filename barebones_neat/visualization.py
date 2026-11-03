import pygame
from .utils import load_font
from collections import defaultdict, deque

def calculate_network_layers(inputs, outputs, connections):
    graph = defaultdict(list)
    for src, dst in connections:
        graph[src].append(dst)
    layers_dict = {node: 0 for node in inputs}
    queue = deque(inputs)
    visited = set(inputs)
    while queue:
        current_node = queue.popleft()
        current_layer = layers_dict[current_node]
        for neighbor in graph[current_node]:
            if neighbor in inputs:
                continue
            new_layer = current_layer + 1
            if neighbor not in layers_dict or new_layer > layers_dict[neighbor]:
                layers_dict[neighbor] = new_layer
                if neighbor not in visited:
                    queue.append(neighbor) 
    all_nodes = set(inputs) | set(outputs) | {u for u, v in connections} | {v for u, v in connections}
    for node in all_nodes:
        if node not in layers_dict:
            layers_dict[node] = 1
    max_hidden_layer = max([layers_dict[n] for n in layers_dict if n not in outputs], default=0)
    for out_node in outputs:
        layers_dict[out_node] = max_hidden_layer + 1
    num_layers = max(layers_dict.values()) + 1
    nested_layers = [[] for _ in range(num_layers)]
    for node, layer_idx in layers_dict.items():
        nested_layers[layer_idx].append(node)
    return nested_layers

class TrainingVisualization:
    def __init__(self, env_width, env_height, screen):
        self.env_width = env_width
        self.env_height = env_height
        self.screen = screen
        self.font = load_font("JetBrainsMono-Regular", 18)

    def draw_ui_shell(self):
        pygame.draw.rect(self.screen, (0, 2, 61), (0, self.env_height, self.screen.get_width(), 2))
        pygame.draw.rect(self.screen, (0, 2, 61), (350, self.env_height, 2, self.screen.get_height() - self.env_height))

    def draw_network(self, compiled_genome, network_surface, hor_padding, vert_padding, method="stretched"):
        input_nodes = []
        output_nodes = []
        for node in compiled_genome.nodes:
            if node[1] == "in":
                input_nodes.append(node[0])
            if node[1] == "out":
                output_nodes.append(node[0])
        layers = calculate_network_layers(input_nodes, output_nodes, compiled_genome.conns)
        node_cords = {} # node id: (x, y)
        amount_layers = len(layers)
        layer_gap = (network_surface.get_width() - hor_padding * 2) / amount_layers - 1
        if method == "stretched":
            for q, layer in enumerate(layers):
                node_gap = (network_surface.get_height() - 2 * vert_padding) / len(layer) - 1
                for i, node in enumerate(layer):
                    x = hor_padding + layer_gap * q
                    y = 0
                    if len(layer) == 1:
                        y = network_surface.get_height() / 2
                    else:
                        y = vert_padding + node_gap * i
                    node_cords[node] = (x, y)
        elif method == "proportional":
            biggest_layer_size = len(max(layers, key=len))
            node_gap = (network_surface.get_height() - 2 * vert_padding) / biggest_layer_size - 1
            for q, layer in enumerate(layers):
                for i, node in enumerate(layer):
                    x = hor_padding + layer_gap * q
                    y = 0
                    if len(layer) == 1:
                        y = network_surface.get_height() / 2
                    else:
                        y = vert_padding + node_gap * i
                    node_cords[node] = (x, y)
        else:
            raise ValueError("This method does not exist, choose either stretched or proportional.")
        for conn in compiled_genome.conns:
            from_node = conn[0]
            to_node = conn[1]
            weight = compiled_genome.weights[conn]
            line_thickness = max(1, int(abs(weight) * 7))
            color = (255, 255, 255) if weight > 0 else (0, 134, 212)
            pygame.draw.line(network_surface, color, node_cords[from_node], node_cords[to_node], line_thickness)
        node_radius = 6
        for node_id, cord in node_cords.items():
            pygame.draw.circle(network_surface, (255, 255, 255), cord, node_radius, width=2)
                    
    def draw_info(self, neat_class, info_surface):
        population_size = neat_class.population_size
        population_size_text = self.font.render(f"population: {population_size}", True, (0, 2, 61))
        amount_species = len(neat_class.species)
        amount_species_text = self.font.render(f"amount species: {amount_species}", True, (0, 2, 61))
        info_surface.blit(population_size_text, (8, 20))
        info_surface.blit(amount_species_text, (8, 60))

    def full_draw(self, compiled_genome, neat_class):
        self.screen.fill((255, 255, 255))
        network_surface = pygame.Surface((350, 100))
        self.draw_network(compiled_genome, network_surface, 10)
        info_surface = pygame.Surface((self.env_width - 3 * 20 - 350, 100))
        self.draw_info(neat_class, info_surface)
        self.draw_ui_shell()

class InferenceVisualization:
    def __init__():
        pass