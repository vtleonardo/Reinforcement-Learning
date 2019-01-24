# Reinforcement-Learning
This repository implements of the deep reinforcement learning algothrims DQN and DRQN in python. The Deep Q-Network (DQN) 
introduced by the article [Human-level control through deep reinforcement learning[1]](#[1]) is an algorithm that applies deep neural networks to the problem of reinforcement learning. 
The reinforcement learning's goal is, basically, how to teach artificial intelligence, called an agent, 
to map the current state in the environment in actions, in such a way to maximize the total reward at end of the task. 
In the DQN case, the states are images from the environment's screen. Therefore, the deep neural network is trained while the agent
interacts with the environment (via reinforcement learning), in such a way to transform these high dimensional data (images) in 
actions (buttons to be pressed). The Deep Recurrent Q-Network (DRQN) is a version of the DQN introduced by the article [Deep recurrent q-learning 
for partially observable mdps[2]](#[2]) that replaces the first fully connected layer of the DQN by one of the type recurrent (LSTM). Thereby, the agent trained by this 
algorithm deals better with partial observable environments, such as the tridimensional maps from ViZDoom.   

## References 
If this repository was useful for your research, please consider citing:
```
@misc{LVTeixeira,
  author = {Leonardo Viana Teixeira},
  title = {Development of an intelligent agent for autonomous exploration of 3D environments via Visual Reinforcement Learning},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/Leonardo-Viana/Reinforcement-Learning},
}
```

## Bibliography

- <a name = "[1]"></a>[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness,
Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg
Ostrovski, et al. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). Nature,
518(7540):529, 2015.
- <a name = "[2]"></a>[2] Matthew Hausknecht and Peter Stone. [Deep recurrent q-learning for partially 
observable mdps](https://arxiv.org/abs/1507.06527). CoRR, abs/1507.06527, 2015.
- <a name = "[3]"></a>[3] Max Lapan. Speeding up dqn on pytorch: how to solve pong in 30 minutes. November 23, 2017. Available at:
https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55. 
Accessed: November 07, 2018.
- <a name = "[4]"></a>[4] Daniel Seita. Frame Skipping and Pre-Processing for Deep Q-Networks on Atari 2600 Games. 
November 25, 2016. Available at: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-
on-atari-2600-games. Accessed: January 04, 2019.
