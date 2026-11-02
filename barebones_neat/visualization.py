import pygame

class Visualization:
    def __init__(self, env_width, env_height, screen):
        self.env_width = env_width
        self.env_height = env_height
        self.screen = screen

    def draw_network(compiled_genome, network_surface):
        pass

    def draw_ui_shell(self):
        pass

    def full_draw(compiled_genome, self):
        network_surface = pygame.Surface((self.env_width - 2 * 20, 100))
        self.draw_network(compiled_genome, network_surface)
        self.draw_ui_shell()