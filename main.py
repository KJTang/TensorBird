import sys
sys.path.append("game/")
sys.path.append("dqn/")

from flappybird import GameApp
from deep_q_learning import DeepQLearning

def main(): 
    # game = GameApp();
    # game.Init();
    # game.AutoGameLoop();
    dqn = DeepQLearning();

if __name__=="__main__":
    main() 