from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2
from gameframe.rect import Rect
from gameframe.event import EventManager

from runtime.game_manager import GameManager
from runtime.bird import Bird
from runtime.reward import Reward
from runtime.pipe import PipeCreator

game_manager = GameManager();
event_manager = EventManager();

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
kBirdOriginPosY = game_manager.ScreenHeight * 0.1;

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

        # fg
        self.foreground = Sprite(kForegroundPath);
        self.foreground.LocalPosition = Vector2(abs(game_manager.ScreenWidth - self.foreground.Width) / 2, game_manager.ScreenHeight - self.foreground.Height);
        self.AddChild(self.foreground);

        # bird
        self.bird = Bird();
        self.bird.LocalPosition = Vector2(kBirdOriginPosX, kBirdOriginPosY);
        self.AddChild(self.bird);

        # reward
        self.reward = Reward();
        self.AddChild(self.reward);

        event_manager.Register("GAME_REWARD", self.OnRewarding);

    def Update(self): 
        pipes = self.pipe_creator.GetPipes()

        # collision check
        if Rect.isOverlapRect(self.bird.Rect, self.foreground.Rect): 
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