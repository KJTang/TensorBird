import os
import pygame

from gameframe.singleton import Singleton
from gameframe.logger import Logger

class SpriteLoader(Singleton):
    def Init(self):
        self._root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../");

    def Load(self, path): 
        return pygame.image.load(self.GetImagePath(path));

    def GetImagePath(self, path): 
        return os.path.join(self._root_path + path);
