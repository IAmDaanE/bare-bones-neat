import pygame

class Visualization:
    def __init__(self, env_width, env_height, screen):
        self.env_width = env_width
        self.env_height = env_height
        self.screen = screen

    def draw_ui_shell(self):
        pass

    def draw_network(compiled_genome, network_surface):
        pass

    def draw_info(neat_class, info_surface):
        population_size = neat_class.population_size

    def full_draw(compiled_genome, neat_class, self):
        network_surface = pygame.Surface((350, 100))
        self.draw_network(compiled_genome, network_surface)
        info_surface = pygame.Surface((self.env_width - 3 * 20 - 350))
        self.draw_info(neat_class, info_surface)
        self.draw_ui_shell()