import time
import pygame

from runtime.game_manager import GameManager
from runtime.play_scene import PlayScene

def main():
    pygame.init();

    # load and set the logo
    logo = pygame.image.load("image/redbird-midflap.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Flappy Bird")
     
    # init game
    game_manager = GameManager();
    screen = pygame.display.set_mode((game_manager.ScreenWidth, game_manager.ScreenHeight))
    clock = pygame.time.Clock()
    scene = PlayScene();

    # define a variable to control the main loop
    running = True
    # main loop
    while running: 
        # event handling, gets all event from the eventqueue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False

        # logic update
        game_manager.Update(scene);

        # render
        game_manager.Render(screen, scene);
        pygame.display.flip(); 

        # fps
        clock.tick(game_manager.TargetFPS);
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()