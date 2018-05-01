from gameframe.singleton import Singleton

from gameframe.sprite import Sprite
from gameframe.vector import Vector2
from gameframe.logger import Logger

kScreenWidth = 336;
kScreenHeight = 448;
kTargetFPS = 60;

class GameManager(Singleton):
    def Init(self):
        self._logger = Logger();

    @property
    def ScreenWidth(self): 
        return kScreenWidth;

    @property
    def ScreenHeight(self): 
        return kScreenHeight;

    @property
    def TargetFPS(self): 
        return kTargetFPS;

    def Log(self, str): 
        self._logger.Log(str);

    def Update(self, sprite): 
        self.__Update(sprite);

    def __Update(self, sprite): 
        sprite.Update();
        for child in sprite.GetChildren(): 
            self.__Update(child);

    def Render(self, screen, sprite): 
        self.__Render(screen, sprite);
        self._logger.Render(screen);

    def __Render(self, screen, sprite): 
        sprite.Render(screen);
        for child in sprite.GetChildren(): 
            self.__Render(screen, child);
