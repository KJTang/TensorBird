from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2

from runtime.game_manager import GameManager
from runtime.bird import Bird
from runtime.pipe import PipeCreator

game_manager = GameManager();

kBackgroundPath = "image/background-black.png";
kForegroundPath = "image/base.png";
kAnimPath = [
    "image/0.png", 
    "image/1.png", 
    "image/2.png", 
    "image/3.png", 
    "image/4.png", 
];

kBirdOriginPosX = game_manager.ScreenWidth * 0.3;
kBirdOriginPosY = game_manager.ScreenHeight * 0.3;

class PlayScene(Sprite):
    def __init__(self):
        Sprite.__init__(self, "");

        # bg        
        background = Sprite(kBackgroundPath);
        self.AddChild(background);

        # pipe 
        pipe_creator = PipeCreator();
        self.AddChild(pipe_creator);

        # fg
        foreground = Sprite(kForegroundPath);
        foreground.LocalPosition = Vector2(abs(game_manager.ScreenWidth - foreground.Width) / 2, game_manager.ScreenHeight - foreground.Height);
        self.AddChild(foreground);

        # bird
        self.bird = Bird();
        self.bird.LocalPosition = Vector2(kBirdOriginPosX, kBirdOriginPosY);
        self.AddChild(self.bird);