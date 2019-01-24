# Examples
The configuration of the scripts can be assigned via terminal commands or written in a file with the same name as the script to be executed with extension .cfg. The parameters that were not sent have their values assigned to default. **Case some terminal command is sent jointly with the execution of the script, the entire configuration of the simulation will be done by the terminal commands, thus, the .cfg file will be ignored and any parameter that was not sent via terminal will have its default value assigned.** If no parameter is sent via terminal, the script will search for a file with the same name as the script with extension .cfg (more details in [Configuration files .CFG](https://github.com/Leonardo-Viana/Reinforcement-Learning/edit/master/docs/cfg.md)). If neither of the options listed above is used, the agent will be trained with all the values assigned as default. In other words, it will be used the hyperparameters shown in the article [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)[[1]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README_eng.md#[1]) for the training of an agent in the game Pong. For more information about each of the configurable parameters available and their default values see the [Documentation](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/doc.md) section or use the following command in the terminal:
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
Instead of every time have to type all these commands, we can use the default values to reduce considerably the number of commands. The default values of the variables can be seen in the [Documentation](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/doc.md). Hence, the same file .cfg can be written as:
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
Another option to configure our scripts is to use the terminal commands directly jointly to the script execution:
````
python Base_agent.py --agent_name "DQNPong30" --num_simul_frames 1000000 --lr 1e-4 --e_min 0.02 --e_lin_decay 100000 --target_update 1000 --num_states_stored 100000 --num_random_play 10000 --optimizer adam  --to_save_episodes True --random_seed 1
````
Both options will train an agent with the hyperparameters specified in the articles above with a random seed fixed in 1 during 1 million frames, as can be seen in the summary presented below.  **Always check the summary presented by the scripts in the begin of the simulation to see if all your desired configuration is set**.

<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/summary-pong30.png" height="70%" width="70%">
</p>

