from gameframe.singleton import Singleton

from gameframe.sprite import Sprite
from gameframe.vector import Vector2
from gameframe.logger import Logger
from gameframe.event import EventManager

kScreenWidth = 336;
kScreenHeight = 448;
kTargetFPS = 60;

event_manager = EventManager();

class GameManager(Singleton):
    _entry_scene = None;

    def Init(self):
        self._logger = Logger();

        # events
        event_manager.Register("GAME_REALLY_QUIT", self.OnGameQuit);
        event_manager.Register("GAME_DIED", self.OnBirdDied);
        event_manager.Register("GAME_REWARD", self.OnRewarding);

    def Restart(self, scene): 
        if not self._entry_scene is None: 
            self._entry_scene.Destroy();
        self._entry_scene = scene;

        self._running = True;
        self._need_restart = False;
        self._reward = 0;
        self._logger.Clear();

    @property
    def ScreenWidth(self): 
        return kScreenWidth;

    @property
    def ScreenHeight(self): 
        return kScreenHeight;

    @property
    def TargetFPS(self): 
        return kTargetFPS;

    @property
    def Running(self): 
        return self._running;

    @property
    def NeedRestart(self): 
        return self._need_restart;

    @property
    def Reward(self): 
        return self._reward;

    def Log(self, str): 
        self._logger.Log(str);

    def Update(self): 
        self.InternalUpdate(self._entry_scene);

    def InternalUpdate(self, sprite): 
        if not sprite.Enable: 
            return;
        sprite.Update();
        for child in sprite.GetChildren(): 
            self.InternalUpdate(child);

    def Render(self, screen): 
        self.InternalRender(screen, self._entry_scene);
        self._logger.Render(screen);

    def InternalRender(self, screen, sprite): 
        if not sprite.Enable: 
            return;
        sprite.Render(screen);
        for child in sprite.GetChildren(): 
            self.InternalRender(screen, child);

    def OnGameQuit(self): 
        self._running = False;
        if self._entry_scene is not None: 
            self._entry_scene.Destroy();

    def OnBirdDied(self): 
        self._need_restart = True;

    def OnRewarding(self): 
        self._reward += 1;