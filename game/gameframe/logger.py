import pygame
from gameframe.singleton import Singleton

kMaxLog = 20;
kLogHeight = 10;
kLogOffsetX = 0;
kLogOffsetY = -20;

class Logger(Singleton):
    def Init(self): 
        pygame.font.init(); 
        self._font = pygame.font.SysFont('Comic Sans MS', 10);
        self._text_list = [];

    def Log(self, str): 
        text = self._font.render("Logger: " + str, False, (255, 255, 255));
        self._text_list.append(text);
        if len(self._text_list) >= kMaxLog: 
            self._text_list.pop(0);

    def Clear(self): 
        self._text_list = [];

    def Update(self): 
        pass

    def Render(self, screen): 
        w, h = pygame.display.get_surface().get_size();
        total = len(self._text_list);
        for i in range(total - 1, -1, -1): 
            screen.blit(self._text_list[i], (kLogOffsetX, kLogOffsetY + h - kLogHeight * (total - i)));
