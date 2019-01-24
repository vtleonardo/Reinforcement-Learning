
The learning parameters used for each of the simulations of the pre-trained weights are described below:

# <a name="common_labyrinth"></a> ViZDoom with the map: Labyrinth
The parameters in common between all the simulations with this map are:
````
env = Doom
config_file_path = ../DoomScenarios/labyrinth.cfg
normalize_input= True
frame_skip = 4 
discount_rate = 0.99
epsilon = 1.0
e_min = 0.1
e_lin_decay = 250000
target_update = 10000
num_random_play = 50000
num_states_stored = 250000
batch_size = 32
optimizer = adam
lr = 1e-4
num_simul_frames = 5000000
steps_save_weights = 50000
to_save_episodes = True
steps_save_episodes = 100
multi_threading = True
random_seed = 1
````

## Weights:grayh4-LSTM-weights-Doom-labyrinth-5000000.h5
Jointly with the parameters in [common](#common_labyrinth) the parameters that are specific for this file are:
````
agent_name = grayh4-LSTM
network_model = DRQN
is_recurrent = True
history_size = 4
input_shape = "84,84,1"
````
**The specific parameters are important to load these pre-trained weights in a simulation**

## Weights:grayh4-weights-Doom-labyrinth-5000000.h5
Jointly with the parameters in [common](#common_labyrinth) the parameters that are specific for this file are:
````
agent_name = grayh4
network_model = DQN
is_recurrent = False
history_size = 4
input_shape = "84,84,1"
````
**The specific parameters are important to load these pre-trained weights in a simulation**

## Weights:grayh8-weights-Doom-labyrinth-5000000.h5
Jointly with the parameters in [common](#common_labyrinth) the parameters that are specific for this file are:
````
agent_name = grayh8
network_model = DQN
is_recurrent = False
history_size = 8
input_shape = "84,84,1"
````
**The specific parameters are important to load these pre-trained weights in a simulation**

## Weights:grayh8-full-reg-weights-Doom-labyrinth-5000000.h5
Jointly with the parameters in [common](#common_labyrinth) the parameters that are specific for this file are:
````
agent_name = grayh8-full-reg
network_model = DQN_regularized
is_recurrent = False
history_size = 8
input_shape = "84,84,1"
````
**The specific parameters are important to load these pre-trained weights in a simulation**

