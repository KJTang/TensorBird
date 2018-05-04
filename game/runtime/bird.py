from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2
from gameframe.event import EventManager

from runtime.game_manager import GameManager
from runtime.sprite_loader import SpriteLoader

game_manager = GameManager();
event_manager = EventManager();
sprite_loader = SpriteLoader();

kBirdAnimPath = [
    sprite_loader.GetImagePath("image/redbird-midflap.png"), 
    sprite_loader.GetImagePath("image/redbird-upflap.png"), 
    sprite_loader.GetImagePath("image/redbird-downflap.png"), 
]

kBirdAnimInterval = 20;
kBirdFlapVelocity = 3;
kBirdAcceleration = -0.1;

class Bird(Animation):
    def __init__(self):
        Animation.__init__(self, kBirdAnimPath[0], kBirdAnimPath, kBirdAnimInterval);
        self._velocity = 0;

    def OnEnable(self): 
        event_manager.Register("GAME_FLAP", self.Flap);

    def OnDisable(self): 
        event_manager.Unregister("GAME_FLAP", self.Flap);

    def Flap(self): 
        self._velocity = kBirdFlapVelocity;

    def Update(self): 
        passedTime = 1;     # passed 1 frame
        lastV = self._velocity;
        curV = self._velocity + kBirdAcceleration * passedTime; 
        offsetY = (lastV + curV) / 2 * passedTime;
        self.LocalPosition = Vector2(self.LocalPosition.x, self.LocalPosition.y - offsetY);
        self._velocity = curV;