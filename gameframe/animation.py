import pygame
from gameframe.vector import Vector2
from gameframe.sprite import Sprite
from gameframe.logger import Logger

class Animation(Sprite): 
    def __init__(self, static_path = "" , anim_paths = None, interval = 1, loop = True):
        Sprite.__init__(self, static_path);
        self._frames = None;
        self._interval = interval if interval >= 1 else 1;
        self._loop = loop;
        self.AddFrames(anim_paths);

        self._timer = 0;
        self._index = 0;

    def AddFrames(self, paths): 
        if (paths is None) or (len(paths) == 0): 
            self._frames = None;
        else: 
            self._frames = [];
            for path in paths: 
                frame = pygame.image.load(path)
                self._frames.append(frame);

    def Update(self): 
        if (self._frames is None) or (len(self._frames) == 0): 
            return;

        self._timer += 1;
        if self._timer < self._interval: 
            return;
        self._timer = 0;

        self._index += 1;
        if self._index >= len(self._frames): 
            self._index = 0;


    def Render(self, screen): 
        if (not self._frames is None) and (len(self._frames) > 0): 
            screen.blit(self._frames[self._index], (self.Position.x, self.Position.y));
