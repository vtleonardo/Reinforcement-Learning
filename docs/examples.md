# Examples
The configuration of the scripts can be assigned via terminal commands or written in a file with the same name as the script to be executed with extension .cfg. The parameters that were not sent have their values assigned to default. **Case some terminal command is sent jointly with the execution of the script, the entire configuration of the simulation will be done by the terminal commands, thus, the .cfg file will be ignored and any parameter that was not sent via terminal will have its default value assigned.** If no parameter is sent via terminal, the script will search for a file with the same name as the script with extension .cfg (more details in [Configuration files .CFG](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/cfg.md)). If neither of the options listed above is used, the agent will be trained with all the values assigned as default. In other words, it will be used the hyperparameters shown in the article [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)[[1]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README_eng.md#[1]) for the training of an agent in the game Pong. For more information about each of the configurable parameters available and their default values see the [Documentation](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/doc.md) section or use the following command in the terminal:
````
python Base_agent.py --help
````
## Pong trained with basic DQN
As the first example, we will train an agent using the hyperparameters specified by the article [Speeding up DQN on PyTorch: how to solve Pong in 30 minutes](https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55)[[3]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README_eng.md#[3]). The hyperparameters that are not specified will be assumed as equal to the hyperparameters shown in the article [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)[[1]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README_eng.md#[1]). The Base_agent.cfg will have to have the following code:
```
agent_mode = train
agent_name = DQNPong30
env = PongNoFrameskip-v4
# Variável exclusiva dos jogos de atari
include_score = False
network_model = DQN
normalize_input = True
is_recurrent = False
frame_skip = 4
num_simul_frames = 1000000
discount_rate = 0.99
lr = 1e-4
epsilon = 1.0
e_min = 0.02
decay_mode = linear
e_lin_decay = 100000
target_update = 1000
num_states_stored = 100000
batch_size = 32
input_shape = "84,84,1"
history_size = 4
num_random_play = 10000
loss_type = huber
optimizer = adam
load_weights = False
steps_save_weights = 50000
path_save_weights = ..\Weights
steps_save_plot = 10000
path_save_plot = ..\Plot
to_save_episodes = True
path_save_episodes = ..\Episodes
silent_mode = False
steps_save_episodes = 50
multi_gpu = False
gpu_device = 0
multi_threading = False
to_render = False
random_seed = 1
```
Instead of every time we have to type all these commands, we can use the default values to reduce considerably the number of commands. The default values of the variables can be seen in the [Documentation](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/doc.md). Hence, the same file .cfg can be written as:
```
agent_name = DQNPong30
num_simul_frames = 1000000
lr = 1e-4
e_min = 0.02
e_lin_decay = 100000
target_update = 1000
num_states_stored = 100000
num_random_play = 10000
optimizer = adam
to_save_episodes = True
random_seed = 1
```
After we save the file, we just need to execute the Base_agent.py script without any argument:
````
python Base_agent.py
````
Another option to configure our scripts is to use the terminal commands directly with the script execution:
````
python Base_agent.py --agent_name "DQNPong30" --num_simul_frames 1000000 --lr 1e-4 --e_min 0.02 --e_lin_decay 100000 --target_update 1000 --num_states_stored 100000 --num_random_play 10000 --optimizer adam  --to_save_episodes True --random_seed 1
````
Both options will train an agent with the hyperparameters specified in the articles above with a random seed fixed in 1 during 1 million frames, as can be seen in the summary presented below.  **Always check the summary presented by the scripts in the begin of the simulation to see if all your desired configuration is right**.

<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/summary-pong30.png" height="90%" width="90%">
</p>

## Traning an agent with ViZDoom
This repository has in its dependencies two maps for the game Doom, labyrinth, and labyrinth test, that have the goal to teach the agent the tridimensional navigation (more detail in the topic [exclusive ViZDoom maps](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/map.md)). To train the agent in the map labyrinth using the DRQN neural network architecture (proposed by [Deep recurrent q-learning for partially observable mdps](https://arxiv.org/abs/1507.06527)[[2]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README_eng.md#[2]))) we can use the following code in the .cfg file:
````
env = Doom
config_file_path = ../DoomScenarios/labyrinth.cfg
agent_name = grayh4-LSTM
network_model = DRQN
is_recurrent = True
optimizer = adam
lr = 1e-4
num_random_play = 50000
num_states_stored = 250000
e_lin_decay = 250000
num_simul_frames = 5000000
steps_save_weights = 50000
to_save_episodes = True
steps_save_episodes = 100
multi_threading = True
````
After we save the file, we just need to execute the Base_agent.py script without any argument:
````
python Base_agent.py
````
Another option to configure our scripts is to use the terminal commands directly with script execution:
````
python Base_agent.py --env Doom --agent_name grayh4-LSTM --config_file_path ../DoomScenarios/labyrinth_test.cfg --network_model DRQN --is_recurrent True --optimizer adam --lr 1e-4 --num_random_play 50000 --num_states_stored 250000 --e_lin_decay 250000 --num_simul_frames 5000000 --steps_save_weights 50000 --to_save_episodes True --steps_save_episodes 100 --multi_threading True
````
With these configurations, the script will train an agent called "gray-LSTM" in the map labyrinth with the DRQN architecture during 5 million frames. This simulation takes advantage of the [multithreading mode](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README_eng.md#--performance) to speed up the simulation. The summary of this training can be seen below:
<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/summary-doomDRQN.png" height="70%" width="70%">
</p>

## Testing a trained agent
The script Base_agent.py has two modes of execution train and test. The train mode is the default, and as the name says it trains an agent using the reinforcement learning. In test mode most of the learning hyperparameters are ignored, the goal of this mode is to test an agent trained. For the correct operation of this mode, it is necessary to specify which neural network architecture was used during training.  The following example shows how to configure the .cfg file to test an agent trained with DRQN in the map called labyrinth, rendering the game to the user:
````
agent_mode = test
env = Doom
config_file_path = ../DoomScenarios/labyrinth.cfg
network_model = DRQN
is_recurrent = True
input_shape = "84,84,1"
history_size = 4
load_weights = True
weights_load_path = ../Weights/Pretrained/Doom/Labyrinth/grayh4-LSTM-weights-Doom-labyrinth-5000000.h5
agent_name = doomh4-lstm-test
to_render = True
to_save_states = False
````
After we save the file, we just need to execute the Base_agent.py script without any argument:
````
python Base_agent.py
````
Another option to configure our scripts is to use the terminal commands directly with script execution:
````
python Base_agent.py --agent_mode test --env Doom --config_file_path ../DoomScenarios/labyrinth.cfg --network_model DRQN --is_recurrent True --input_shape "84,84,1" --history_size 4 --load_weights True --weights_load_path ../Weights/Pretrained/Doom/Labyrinth/grayh4-LSTM-weights-Doom-labyrinth-5000000.h5 --agent_name doomh4-lstm-test --to_render True --to_save_states False
````

The simulation summary can be seen below.
<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/summary-doomDRQN-test.png" height="70%" width="70%">
</p>

## 
