import pygame
from importlib import resources
import io

def load_font(font_name, font_size):
    asset_path = resources.files("barebones_ml").joinpath(f"assets/{font_name}.ttf")
    font_bytes = asset_path.read_bytes()
    font_stream = io.BytesIO(font_bytes)
    return pygame.font.Font(font_stream, font_size)