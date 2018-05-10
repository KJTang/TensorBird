from gameframe.sprite import Sprite
from gameframe.vector import Vector2
from gameframe.rect import Rect

from runtime.game_manager import GameManager
from runtime.sprite_loader import SpriteLoader

game_manager = GameManager();
sprite_loader = SpriteLoader();

kForegroundPath = sprite_loader.GetImagePath("image/base.png");
kFloorSpeed = -3;

class Floor(Sprite):
    _floor_width = 0;
    _floor_height = 0;

    def __init__(self):
        Sprite.__init__(self, "");

        self._floor_a = Sprite(kForegroundPath);
        self._floor_b = Sprite(kForegroundPath);
        self.AddChild(self._floor_a);
        self.AddChild(self._floor_b);

        self._floor_width = self._floor_a.Width;
        self._floor_height = self._floor_a.Height;

        self._floor_a.LocalPosition = Vector2(0, game_manager.ScreenHeight - self._floor_height);
        self._floor_b.LocalPosition = Vector2(self._floor_width, game_manager.ScreenHeight - self._floor_height);

    def Update(self): 
        self._floor_a.LocalPosition = Vector2(self._floor_a.LocalPosition.x + kFloorSpeed, self._floor_a.LocalPosition.y);
        self._floor_b.LocalPosition = Vector2(self._floor_b.LocalPosition.x + kFloorSpeed, self._floor_b.LocalPosition.y);

        if self._floor_a.LocalPosition.x <= -self._floor_width: 
            self._floor_a.LocalPosition = Vector2(self._floor_width, self._floor_a.LocalPosition.y);
        if self._floor_b.LocalPosition.x <= -self._floor_width: 
            self._floor_b.LocalPosition = Vector2(self._floor_width, self._floor_b.LocalPosition.y);

    @property
    def Rect(self): 
        if self._floor_a.Position.x < self._floor_b.Position.x: 
            self._rect = Rect(self._floor_a.Position.x, self._floor_a.Position.y, self._floor_width * 2, self._floor_height);
        else:
            self._rect = Rect(self._floor_b.Position.x, self._floor_b.Position.y, self._floor_width * 2, self._floor_height);
        return self._rect;