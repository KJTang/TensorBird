import time
import pygame

from gameframe.event import EventManager
from runtime.game_manager import GameManager
from runtime.sprite_loader import SpriteLoader
from runtime.test_scene import TestScene

game_manager = GameManager();
event_manager = EventManager();
sprite_loader = SpriteLoader();

kLOGOPath = sprite_loader.GetImagePath("image/redbird-midflap.png");
kTitleName = "Flappy Bird Test";

class GameApp():
    def __init__(self):
        pass

    def Init(self): 
        pygame.init();

        # load and set the logo
        logo = pygame.image.load(kLOGOPath)
        pygame.display.set_icon(logo)
        pygame.display.set_caption(kTitleName)
         
        # init game
        self._screen = pygame.display.set_mode((game_manager.ScreenWidth, game_manager.ScreenHeight))
        self._clock = pygame.time.Clock()
        # entry scene
        game_manager.Restart(TestScene());

    def AutoGameLoop(self): 
        while game_manager.Running: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    event_manager.Dispatch("GAME_REALLY_QUIT");
                # trigger flap
                if event.type == pygame.MOUSEBUTTONDOWN: 
                    event_manager.Dispatch("GAME_FLAP");
                elif event.type == pygame.KEYDOWN:
                    event_manager.Dispatch("GAME_FLAP");

            # logic update
            game_manager.Update();

            # render
            self._screen.fill((0, 0, 0));     # clear
            game_manager.Render(self._screen);
            pygame.display.flip(); 

            # fps
            self._clock.tick(game_manager.TargetFPS);

            if game_manager.NeedRestart: 
                game_manager.Restart(TestScene());
        

    def ManualGameLoop(self, actions = [1, 0]): 
        image_data = None;
        terminal = False;
        reward = game_manager.Reward;

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                event_manager.Dispatch("GAME_REALLY_QUIT");

        # actions[0] == 1: do nothing
        # actions[1] == 1: flap the bird
        if actions[1] == 1:
            event_manager.Dispatch("GAME_FLAP");

        # logic update
        game_manager.Update();

        # render
        self._screen.fill((0, 0, 0));     # clear
        game_manager.Render(self._screen);
        pygame.display.flip(); 

        # fps
        self._clock.tick(game_manager.TargetFPS);

        image_data = pygame.surfarray.array3d(pygame.display.get_surface());
        # normalize reward
        if reward > game_manager.Reward: 
            reward = 1.0;
        elif game_manager.NeedRestart: 
            reward = -1.0;
        else: 
            reward = 0.0;
        if game_manager.NeedRestart: 
            terminal = True;
            game_manager.Restart(TestScene());

        return image_data, reward, terminal



def main(): 
    game = GameApp();
    game.Init();
    game.AutoGameLoop();
     

# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()