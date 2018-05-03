import pygame
from gameframe.sprite import Sprite
from gameframe.animation import Animation
from gameframe.vector import Vector2
from gameframe.event import EventManager

from runtime.game_manager import GameManager
from runtime.sprite_loader import SpriteLoader

game_manager = GameManager();
event_manager = EventManager();
sprite_loader = SpriteLoader();

kRewardPath = [
    sprite_loader.GetImagePath("image/0.png"), 
    sprite_loader.GetImagePath("image/1.png"), 
    sprite_loader.GetImagePath("image/2.png"), 
    sprite_loader.GetImagePath("image/3.png"), 
    sprite_loader.GetImagePath("image/4.png"), 
    sprite_loader.GetImagePath("image/5.png"), 
    sprite_loader.GetImagePath("image/6.png"), 
    sprite_loader.GetImagePath("image/7.png"), 
    sprite_loader.GetImagePath("image/8.png"), 
    sprite_loader.GetImagePath("image/9.png"), 
];

kRewardLen = 3;
kRewardOffsetX = 10;
kRewardOffsetY = 10;

class Reward(Sprite):
    def __init__(self, reward = 0):
        Sprite.__init__(self, kRewardPath[0]);
        self._reward = reward;
        self._reward_list = [];
        self._reward_pic = [];
        for path in kRewardPath: 
            self._reward_pic.append(pygame.image.load(path));

        self.SetReward(reward);

    def SetReward(self, reward): 
        self._reward = int(reward);
        self._reward_list.clear();
        while reward > 0: 
            self._reward_list.append(int(reward % 10));
            reward = int(reward / 10);

        while len(self._reward_list) < kRewardLen: 
            self._reward_list.append(0);

        self._reward_list.reverse();

    def Render(self, screen): 
        total = len(self._reward_list);
        for i in range(0, total): 
            screen.blit(self._reward_pic[self._reward_list[i]], (kRewardOffsetX + self.Width * i, kRewardOffsetY));