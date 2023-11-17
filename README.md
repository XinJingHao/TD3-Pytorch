# TD3-Pytorch

**A clean and robust Pytorch implementation of TD3 on continuous action space.**

<img src="https://github.com/XinJingHao/TD3-Pytorch/blob/main/images/Render_BWHV3.gif" width="80%" height="auto">  | <img src="https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/render_gif/lldcV2.gif" width="80%" height="auto">
:-----------------------:|:-----------------------:|

**Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**

## Dependencies
```python
gymnasium==0.29.1
box2d-py==2.3.5
numpy==1.26.1
pytorch==2.1.0
tensorboard==2.15.1
packaging==23.2

python==3.11.5
```

## How to use my code
### Train from scratch
```bash
python main.py
```
where the default enviroment is Pendulum-v1.  

### Play with trained model
```bash
python main.py --EnvIdex 0 --write False --render True --Loadmodel True --ModelIdex 10
```
which will render the 'Pendulum-v1'.  

### Change Enviroment
If you want to train on different enviroments, just run 
```bash
python main.py --EnvIdex 1
```
The ```--EnvIdex``` can be set to be 0~5, where
```bash
'--EnvIdex 0' for 'Pendulum-v0'  
'--EnvIdex 1' for 'LunarLanderContinuous-v2'  
'--EnvIdex 2' for 'Humanoid-v2'  
'--EnvIdex 3' for 'HalfCheetah-v2'  
'--EnvIdex 4' for 'BipedalWalker-v3'  
'--EnvIdex 5' for 'BipedalWalkerHardcore-v3' 
```

Note: if you want train on BipedalWalker-v3, BipedalWalkerHardcore-v3, or LunarLanderContinuous-v2, you need to install [box2d-py](https://gymnasium.farama.org/environments/box2d/) first.

if you want train on Humanoid-v2 or HalfCheetah-v2, you need to install [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) first.

### Visualize the training curve
You can use the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to visualize the training curve. History training curve is saved at '\runs'

### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'

### Reference
TD3: Fujimoto S , Hoof H V , Meger D . Addressing Function Approximation Error in Actor-Critic Methods[J]. 2018.

## Training Curves
<img src="https://github.com/XinJingHao/TD3-Pytorch/blob/main/images/TD3results.png" width=700>
All the experiments are trained with same hyperparameters (see main.py). 
