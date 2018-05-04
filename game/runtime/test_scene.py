import random

from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2
from gameframe.rect import Rect
from gameframe.event import EventManager

from runtime.game_manager import GameManager
from runtime.sprite_loader import SpriteLoader
from runtime.reward import Reward

game_manager = GameManager();
event_manager = EventManager();
sprite_loader = SpriteLoader();

kBirdPath = sprite_loader.GetImagePath("image/background-white.png");
# kBirdInterval = [30, 300];
kBirdInterval = 60;

kBirdOriginPosX = game_manager.ScreenWidth * 0.5;
kBirdOriginPosY = game_manager.ScreenHeight * 0.5;

class TestScene(Sprite):
    def __init__(self):
        Sprite.__init__(self, "");

        # bg        
        self.bird = None;

        # reward
        self.reward = Reward();
        self.AddChild(self.reward);

        self.timer = 0;
        # self.cur_interval = random.randrange(kBirdInterval[0], kBirdInterval[1]);
        self.cur_interval = kBirdInterval;

    def OnEnable(self): 
        event_manager.Register("GAME_REWARD", self.OnRewarding);
        event_manager.Register("GAME_FLAP", self.OnFlapping);

    def OnDisable(self): 
        event_manager.Unregister("GAME_REWARD", self.OnRewarding);
        event_manager.Unregister("GAME_FLAP", self.OnFlapping);

    def Update(self): 
        self.timer += 1;
        if self.timer >= self.cur_interval: 
            if self.bird is None: 
                self.timer = 0;
                # self.cur_interval = random.randrange(kBirdInterval[0], kBirdInterval[1]);
                self.cur_interval = kBirdInterval;

                self.bird = Sprite(kBirdPath);
                # self.bird.LocalPosition = Vector2(kBirdOriginPosX, kBirdOriginPosY);
                self.AddChild(self.bird);
            else: # didnt flap
                event_manager.Dispatch("GAME_DIED");
                

    def OnRewarding(self): 
        self.reward.SetReward(game_manager.Reward);

    def OnFlapping(self): 
        if self.bird is None: 
            event_manager.Dispatch("GAME_DIED");
        else: 
            self.RemoveChild(self.bird);
            self.bird = None;
            event_manager.Dispatch("GAME_REWARD");
