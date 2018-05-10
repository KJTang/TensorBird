from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2
from gameframe.rect import Rect
from gameframe.event import EventManager

from runtime.game_manager import GameManager
from runtime.sprite_loader import SpriteLoader
from runtime.bird import Bird
from runtime.reward import Reward
from runtime.pipe import PipeCreator
from runtime.floor import Floor

game_manager = GameManager();
event_manager = EventManager();
sprite_loader = SpriteLoader();

kBackgroundPath = sprite_loader.GetImagePath("image/background-black.png");

kBirdOriginPosX = game_manager.ScreenWidth * 0.3;
kBirdOriginPosY = game_manager.ScreenHeight * 0.3;

class PlayScene(Sprite):
    def __init__(self):
        Sprite.__init__(self, "");

        # bg        
        background = Sprite(kBackgroundPath);
        self.AddChild(background);

        # pipe 
        self.pipe_creator = PipeCreator();
        self.AddChild(self.pipe_creator);
        self.lastPipe = None;

        # floor
        self.floor = Floor();
        self.AddChild(self.floor);

        # bird
        self.bird = Bird();
        self.bird.LocalPosition = Vector2(kBirdOriginPosX, kBirdOriginPosY);
        self.AddChild(self.bird);

        # reward
        self.reward = Reward();
        self.AddChild(self.reward);

    def OnEnable(self): 
        event_manager.Register("GAME_REWARD", self.OnRewarding);

    def OnDisable(self): 
        event_manager.Unregister("GAME_REWARD", self.OnRewarding);

    def Update(self): 
        pipes = self.pipe_creator.GetPipes()

        # collision check
        if Rect.isOverlapRect(self.bird.Rect, self.floor.Rect): 
            event_manager.Dispatch("GAME_DIED");
        elif self.bird.Position.y <= 0: 
            event_manager.Dispatch("GAME_DIED");
        else: 
            for pipe in pipes: 
                if Rect.isOverlapRect(self.bird.Rect, pipe.Rect): 
                    event_manager.Dispatch("GAME_DIED");
                    break;
            

        # collect reward
        for pipe in pipes: 
            if self.bird.Position.x >= pipe.Position.x and pipe.IsReversed(): # we only use one side pipes
                if pipe != self.lastPipe: 
                    self.lastPipe = pipe;
                    event_manager.Dispatch("GAME_REWARD");
                    break;


    def OnRewarding(self): 
        self.reward.SetReward(game_manager.Reward);