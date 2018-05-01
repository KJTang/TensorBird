from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2
from runtime.game_manager import GameManager

game_manager = GameManager();

kBirdAnimPath = [
    "image/redbird-midflap.png", 
    "image/redbird-upflap.png", 
    "image/redbird-downflap.png", 
]

kBirdAnimInterval = 20;

class Bird(Animation):
    def __init__(self):
        Animation.__init__(self, kBirdAnimPath, kBirdAnimInterval);

    def Flap(): 
        pass