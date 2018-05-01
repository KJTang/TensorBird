import random
import pygame
from gameframe.singleton import Singleton
from gameframe.sprite import Sprite
from gameframe.vector import Vector2
from runtime.game_manager import GameManager

game_manager = GameManager();

kPipePath = "image/pipe-green.png"; 
kPipeSpeed = -3;
kPipeGapHeight = 120;
kPipeMaxOffset = 120;
kPipeCreateInterval = 100;   # frames

class Pipe(Sprite):
    def __init__(self):
        Sprite.__init__(self, kPipePath);

    def SetupPipe(self, reverse, offset): 
        if reverse: 
            self._img = pygame.transform.rotate(self._img, 180);
            self.LocalPosition = Vector2.Add(Vector2(0, game_manager.ScreenHeight / 2 - self.Height - kPipeGapHeight), offset);
        else:
            self.LocalPosition = Vector2.Add(Vector2(0, game_manager.ScreenHeight / 2), offset);


class PipeCreator(Sprite): 
    def __init__(self, interval = kPipeCreateInterval): 
        Sprite.__init__(self, "");
        self._interval = interval;
        self._timer = 0;
        self._pipes = [];
        self._created = 0;  # debug

    def Update(self): 
        # create 
        self._timer += 1;
        if self._timer >= self._interval: 
            self._timer = 0; 
            self._created += 1;
            # game_manager.Log("create " + str(self._created));

            offsetY = (random.random() - 0.5) * kPipeMaxOffset;
            
            pipe_up = Pipe();
            pipe_up.SetupPipe(True, Vector2(game_manager.ScreenWidth, offsetY));
            self.AddChild(pipe_up);
            self.AddPipe(pipe_up);

            pipe_down = Pipe();
            pipe_down.SetupPipe(False, Vector2(game_manager.ScreenWidth, offsetY));
            self.AddChild(pipe_down); 
            self.AddPipe(pipe_down);

        # move
        for pipe in self._pipes: 
            pipe.LocalPosition = Vector2(pipe.LocalPosition.x + kPipeSpeed, pipe.LocalPosition.y);

        # destroy
        to_remove = [];
        for pipe in self._pipes: 
            if pipe.LocalPosition.x < 0: 
                to_remove.append(pipe);
        for pipe in to_remove: 
            self.RemovePipe(pipe);
            self.RemoveChild(pipe);

    def AddPipe(self, pipe): 
        self._pipes.append(pipe);

    def RemovePipe(self, pipe): 
        for i in range(0, len(self._pipes)): 
            if self._pipes[i] == pipe: 
                self._pipes.pop(i);
                break;

    def GetPipes(self, pipe): 
        return self._pipes;
