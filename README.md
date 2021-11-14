# TD3-Pytorch
A clean and robust Pytorch implementation of TD3 on continuous action space. Here is the result:  
<img src="https://github.com/XinJingHao/TD3-Pytorch/blob/main/TD3results.png" width=700>

All the experiments are trained with same hyperparameters. 

## Dependencies
gym==0.18.3  
box2d==2.3.10  
numpy==1.21.2  
pytorch==1.8.1  

## How to use my code
### Train from scratch
run **'python main.py'**, where the default enviroment is Pendulum-v0.  
### Change Enviroment
If you want to train on different enviroments, just run **'python main.py --EnvIdex 1'**.  
The --EnvIdex can be set to be 0~5, where   
'--EnvIdex 0' for 'Pendulum-v0'  
'--EnvIdex 1' for 'LunarLanderContinuous-v2'  
'--EnvIdex 2' for 'Humanoid-v2'  
'--EnvIdex 3' for 'HalfCheetah-v2'  
'--EnvIdex 4' for 'BipedalWalker-v3'  
'--EnvIdex 5' for 'BipedalWalkerHardcore-v3' 

P.S. if you want train on 'Humanoid-v2' or 'HalfCheetah-v2', you need to install **MuJoCo** first.
### Play with trained model
run **'python main.py --EnvIdex 0 --write False --render True --Loadmodel True --ModelIdex 30000'**, which will render the 'Pendulum-v0'.  
### Visualize the training curve
You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'
### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'
### Reference
TD3: Fujimoto S , Hoof H V , Meger D . Addressing Function Approximation Error in Actor-Critic Methods[J]. 2018.
