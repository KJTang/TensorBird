## TensorBird
<img src="display.gif" width = "300" alt="flappy" align=center />

### Intro

use deep-q-learning algorithm to train agent to play flappy -bird game.   
coding by python3, used frameworks:   

* tensorflow  
* opencv  
* pygame

### How to run
cd into project directory, type command in terminal: 
```  
python dqn.py
```

### How to re-training
edit dqn.py:   
comment code:   
```  
kEpsilonInit = 0.0001; 
```
rewrite it:   
```  
kEpsilonInit = 0.1000; 
```
re-run to start training. 

### Ref
[[1] Playing FlappyBird with Deep Reinforcement Learning](http://101.110.118.33/cs231n.stanford.edu/reports/2016/pdfs/111_Report.pdf)  
[[2] Github: DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)  

