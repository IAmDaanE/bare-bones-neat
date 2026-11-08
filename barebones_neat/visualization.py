import pygame
from .utils import load_font
from collections import defaultdict, deque

def calculate_network_layers(inputs, outputs, connections):
    graph = defaultdict(list)
    for src, dst in connections:
        graph[src].append(dst)
    layers_dict = {node: 0 for node in inputs}
    queue = deque(inputs)
    processed = set()
    while queue:
        current_node = queue.popleft()
        if current_node in processed:
            continue
        processed.add(current_node)
        current_layer = layers_dict[current_node]
        for neighbor in graph[current_node]:
            if neighbor in inputs:
                continue
            new_layer = current_layer + 1
            if neighbor not in layers_dict or new_layer > layers_dict[neighbor]:
                layers_dict[neighbor] = new_layer
            if neighbor not in processed:
                queue.append(neighbor)       
    all_nodes = set(inputs) | set(outputs) | {u for u, v in connections} | {v for u, v in connections}
    for node in all_nodes:
        if node not in layers_dict:
            layers_dict[node] = 1
    max_hidden_layer = max([layers_dict[n] for n in layers_dict if n not in outputs], default=0)
    for out_node in outputs:
        layers_dict[out_node] = max_hidden_layer + 1
    unique_assigned_layers = sorted(list(set(layers_dict.values())))    
    layer_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_assigned_layers)}    
    for node in layers_dict:
        layers_dict[node] = layer_mapping[layers_dict[node]]        
    num_layers = len(unique_assigned_layers)
    nested_layers = [[] for _ in range(num_layers)]
    for node, layer_idx in layers_dict.items():
        nested_layers[layer_idx].append(node)
    return nested_layers

class TrainingVisualization:
    def __init__(self, env_width, env_height, screen, input_names, output_names):
        self.env_width = env_width
        self.env_height = env_height
        self.screen = screen
        self.font = load_font("JetBrainsMono-Regular", 16)
        self.small_font = load_font("JetBrainsMono-Regular", 10)
        self.input_names = input_names
        self.output_names = output_names

    def draw_ui_shell(self):
        pygame.draw.rect(self.screen, (0, 2, 61), (0, self.env_height, self.screen.get_width(), 2))
        pygame.draw.rect(self.screen, (0, 2, 61), (420, self.env_height, 2, self.screen.get_height() - self.env_height))

    def draw_network(self, compiled_genome, network_surface, hor_padding, vert_padding, left_names_offset, right_names_offset, vert_names_offset, method="stretched", weight_mult=7, max_line_thickness=35):
        width = network_surface.get_width()
        height = network_surface.get_height()
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
        layer_gap = (width - hor_padding * 2) / (amount_layers - 1)
        if method == "stretched":
            for q, layer in enumerate(layers):
                if len(layer) > 1:
                    node_gap = (height - 2 * vert_padding) / (len(layer) - 1) 
                for i, node in enumerate(layer):
                    x = hor_padding + layer_gap * q
                    y = 0
                    if len(layer) == 1:
                        y = height / 2
                    else:
                        y = vert_padding + node_gap * i
                    node_cords[node] = (x, y)
        elif method == "proportional":
            biggest_layer_size = len(max(layers, key=len))
            node_gap = (height - 2 * vert_padding) / (biggest_layer_size - 1)
            for q, layer in enumerate(layers):
                for i, node in enumerate(layer):
                    x = hor_padding + layer_gap * q
                    y = 0
                    if len(layer) == 1:
                        y = height / 2
                    else:
                        y = vert_padding + node_gap * i
                    node_cords[node] = (x, y)
        else:
            raise ValueError("This method does not exist, choose either stretched or proportional.")
        for conn in compiled_genome.conns:
            from_node = conn[0]
            to_node = conn[1]
            weight = compiled_genome.weights[conn]
            line_thickness = min(max(1, int(abs(weight * weight_mult))), max_line_thickness)
            color = (0, 2, 61) if weight < 0 else (0, 134, 212)
            pygame.draw.line(network_surface, color, node_cords[from_node], node_cords[to_node], line_thickness)
        node_radius = 6
        for node_id, cord in node_cords.items():
            pygame.draw.circle(network_surface, (0, 0, 0), cord, node_radius)
        for i, text in enumerate(self.input_names):
            text = self.small_font.render(text, True, (0, 2, 61))
            network_surface.blit(text, (node_cords[input_nodes[i]][0] - left_names_offset, node_cords[input_nodes[i]][1] - vert_names_offset))
        for i, text in enumerate(self.output_names):
            text = self.small_font.render(text, True, (0, 2, 61))
            network_surface.blit(text, (node_cords[output_nodes[i]][0] + right_names_offset, node_cords[output_nodes[i]][1] - vert_names_offset))
                    
    def draw_info(self, neat_class, info_surface):
        population_size = neat_class.population_size
        population_size_text = self.font.render(f"Population:  {population_size}", True, (0, 134, 212))
        amount_species = len(neat_class.species)
        amount_species_text = self.font.render(f"Amount Species:  {amount_species}", True, (0, 134, 212))
        similarity_treshold = neat_class.similarity_treshold
        similarity_treshold_text = self.font.render(f"Sim Treshold:  {similarity_treshold}", True, (0, 134, 212))
        mutation_power = neat_class.mutation_power
        mutation_power_text = self.font.render(f"Mutation:  {mutation_power}", True, (0, 134, 212))
        new_node_chance  = neat_class.chances["new_node"]
        new_node_chance_text = self.font.render(f"Node Chance:  {new_node_chance}%", True, (0, 134, 212))
        side_padding = 6
        between_padding = 27
        info_surface.blit(population_size_text, (8, side_padding))
        info_surface.blit(amount_species_text, (8, side_padding + between_padding))
        info_surface.blit(similarity_treshold_text, (8, side_padding + between_padding * 2))
        info_surface.blit(mutation_power_text, (8, side_padding + between_padding * 3))
        info_surface.blit(new_node_chance_text, (8, side_padding + between_padding * 4))

    def full_draw(self, compiled_genome, neat_class, network_hor_padding=60, network_vert_padding=20, left_names_offset=50, right_names_offset=8, vert_names_offset=5, network_weight_mult=7):
        self.screen.fill((255, 255, 255))
        network_surface = pygame.Surface((420, 140))
        network_surface.fill((255, 255, 255))
        self.draw_network(compiled_genome, network_surface, network_hor_padding, network_vert_padding, left_names_offset, right_names_offset, vert_names_offset, method="stretched", weight_mult=network_weight_mult)
        info_surface = pygame.Surface((self.env_width - 420, 140))
        info_surface.fill((255, 255, 255))
        self.draw_info(neat_class, info_surface)
        self.screen.blit(network_surface, (0, self.env_height))
        self.screen.blit(info_surface, (420, self.env_height))
        self.draw_ui_shell()

class InferenceVisualization:
    def __init__(self, env_width, env_height, screen, input_names, output_names):
        self.env_width = env_width
        self.env_height = env_height
        self.screen = screen
        self.font = load_font("JetBrainsMono-Regular", 16)
        self.small_font = load_font("JetBrainsMono-Regular", 10)
        self.input_names = input_names
        self.output_names = output_names

    def draw_ui_shell(self):
        pygame.draw.rect(self.screen, (0, 2, 61), (0, self.env_height, self.screen.get_width(), 2))
        pygame.draw.rect(self.screen, (0, 2, 61), (420, self.env_height, 2, self.screen.get_height() - self.env_height))

    def draw_network(self, compiled_genome, network_surface, hor_padding, vert_padding, left_names_offset, right_names_offset, vert_names_offset, method="stretched", weight_mult=7, max_line_thickness=35):
        width = network_surface.get_width()
        height = network_surface.get_height()
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
        layer_gap = (width - hor_padding * 2) / (amount_layers - 1)
        if method == "stretched":
            for q, layer in enumerate(layers):
                if len(layer) > 1:
                    node_gap = (height - 2 * vert_padding) / (len(layer) - 1) 
                for i, node in enumerate(layer):
                    x = hor_padding + layer_gap * q
                    y = 0
                    if len(layer) == 1:
                        y = height / 2
                    else:
                        y = vert_padding + node_gap * i
                    node_cords[node] = (x, y)
        elif method == "proportional":
            biggest_layer_size = len(max(layers, key=len))
            node_gap = (height - 2 * vert_padding) / (biggest_layer_size - 1)
            for q, layer in enumerate(layers):
                for i, node in enumerate(layer):
                    x = hor_padding + layer_gap * q
                    y = 0
                    if len(layer) == 1:
                        y = height / 2
                    else:
                        y = vert_padding + node_gap * i
                    node_cords[node] = (x, y)
        else:
            raise ValueError("This method does not exist, choose either stretched or proportional.")
        for conn in compiled_genome.conns:
            from_node = conn[0]
            to_node = conn[1]
            weight = compiled_genome.weights[(from_node, to_node)]
            line_thickness = min(max(1, int(abs(compiled_genome.conn_values[(from_node, to_node)]) * weight_mult)), max_line_thickness)
            color = (0, 2, 61) if weight < 0 else (0, 134, 212)
            pygame.draw.line(network_surface, color, node_cords[from_node], node_cords[to_node], line_thickness)
        node_radius = 6
        for node_id, cord in node_cords.items():
            pygame.draw.circle(network_surface, (0, 0, 0), cord, node_radius)
        for i, text in enumerate(self.input_names):
            text = self.small_font.render(text, True, (0, 2, 61))
            network_surface.blit(text, (node_cords[input_nodes[i]][0] - left_names_offset, node_cords[input_nodes[i]][1] - vert_names_offset))
        for i, text in enumerate(self.output_names):
            text = self.small_font.render(text, True, (0, 2, 61))
            network_surface.blit(text, (node_cords[output_nodes[i]][0] + right_names_offset, node_cords[output_nodes[i]][1] - vert_names_offset))
                    
    def draw_info(self, compiled_genome, info_surface):
        amount_nodes = len(compiled_genome.nodes)
        amount_nodes_text = self.font.render(f"Amount Nodes:  {amount_nodes}", True, (0, 2, 61))
        input_size = compiled_genome.input_size
        input_size_text = self.font.render(f"Input Size:  {input_size}", True, (0, 2, 61))
        output_size = compiled_genome.output_size
        output_size_text = self.font.render(f"Output Size:  {output_size}", True, (0, 2, 61))
        amount_conns = len(compiled_genome.conns)
        amount_conns_text = self.font.render(f"Amount Conns:  {amount_conns}", True, (0, 2, 61))
        use_softmax  = compiled_genome.use_softmax
        use_softmax_text = self.font.render(f"Use Softmax:  {use_softmax}", True, (0, 2, 61))
        side_padding = 6
        between_padding = 27
        info_surface.blit(amount_nodes_text, (8, side_padding))
        info_surface.blit(input_size_text, (8, side_padding + between_padding))
        info_surface.blit(output_size_text, (8, side_padding + between_padding * 2))
        info_surface.blit(amount_conns_text, (8, side_padding + between_padding * 3))
        info_surface.blit(use_softmax_text, (8, side_padding + between_padding * 4))

    def full_draw(self, compiled_genome, neat_class, network_hor_padding=60, network_vert_padding=20, left_names_offset=50, right_names_offset=8, vert_names_offset=5, network_weight_mult=7, max_line_thickness=35):
        self.screen.fill((255, 255, 255))
        network_surface = pygame.Surface((420, 140))
        network_surface.fill((255, 255, 255))
        self.draw_network(compiled_genome, network_surface, network_hor_padding, network_vert_padding, left_names_offset, right_names_offset, vert_names_offset, method="stretched", weight_mult=network_weight_mult, max_line_thickness=max_line_thickness)
        info_surface = pygame.Surface((self.env_width - 420, 140))
        info_surface.fill((255, 255, 255))
        self.draw_info(compiled_genome, info_surface)
        self.screen.blit(network_surface, (0, self.env_height))
        self.screen.blit(info_surface, (420, self.env_height))
        self.draw_ui_shell()