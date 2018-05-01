from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2

from runtime.game_manager import GameManager
from runtime.bird import Bird

game_manager = GameManager();

kBGPath = "image/background-black.png";
kBGPrePath = "image/base.png";
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
        backgroud = Sprite(kBGPath);
        self.AddChild(backgroud);
        
        backgroud_pre = Sprite(kBGPrePath);
        backgroud_pre.LocalPosition = Vector2(abs(game_manager.ScreenWidth - backgroud_pre.Width) / 2, game_manager.ScreenHeight - backgroud_pre.Height);
        self.AddChild(backgroud_pre);

        # bird
        self.bird = Bird();
        self.bird.LocalPosition = Vector2(kBirdOriginPosX, kBirdOriginPosY);
        self.AddChild(self.bird);

        # pipe 
        # TODO
