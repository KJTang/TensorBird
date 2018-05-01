import time
import pygame

from gameframe.event import EventManager
from runtime.game_manager import GameManager
from runtime.play_scene import PlayScene

game_manager = GameManager();
event_manager = EventManager();

def main():
    pygame.init();

    # load and set the logo
    logo = pygame.image.load("image/redbird-midflap.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Flappy Bird")
     
    # init game
    screen = pygame.display.set_mode((game_manager.ScreenWidth, game_manager.ScreenHeight))
    clock = pygame.time.Clock()
    # entry scene
    game_manager.Restart(PlayScene());

    # main loop
    while game_manager.Running: 
        # event handling, gets all event from the eventqueue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                event_manager.Dispatch("GAME_REALLY_QUIT");

            # trigger flap
            if event.type == pygame.MOUSEBUTTONDOWN: 
                event_manager.Dispatch("GAME_FLAP");
            if event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_RETURN:
                event_manager.Dispatch("GAME_FLAP");

        # logic update
        game_manager.Update();

        # render
        screen.fill((0, 0, 0));     # clear
        game_manager.Render(screen);
        pygame.display.flip(); 

        # fps
        clock.tick(game_manager.TargetFPS);

        if game_manager.NeedRestart: 
            game_manager.Restart(PlayScene());
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()