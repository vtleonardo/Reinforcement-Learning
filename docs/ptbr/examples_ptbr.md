# Exemplos
A seguir serão apresentados alguns exemplos. Todos os parâmetros podem ser passados via comandos de terminal na execução do script ou via arquivo .cfg (como visto na sessão [Utilização](https://github.com/Leonardo-Viana/Reinforcement-Learning#utiliza%C3%A7%C3%A3o)). Relembrando que os parâmetros não configurados possuem seus valores iguais ao default. Para mais informações sobre cada opção disponível e seus valores default verificar o [DOC](/docs/ptbr/doc_ptbr.md) ou utilizar o comando de terminal:
````
python Base_agent.py --help
````
## Pong treinado com DQN básico
Como primeiro exemplo treinaremos um agente utilizando os hiperparâmetros especificados pelo excelente artigo [Speeding up DQN on PyTorch: how to solve Pong in 30 minutes](https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55)[[3]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README.md#[3])(Os hiperparâmetros não especificados nesse artigo serão assumidos como sendo iguais ao de [[1]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README.md#[1])). O arquivo Base_agent.cfg deverá possuir :
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
Ao invés de toda vez termos que digitar todos esses comandos, podemos utilizar os valores default (padrão) das variáveis para diminuir consideravelmente o número de comandos. Os valores padrões das variáveis podem ser visto no [DOC](/docs/ptbr/doc_ptbr.md). Desta forma o mesmo arquivo .cfg pode ser escrito da seguinte forma:
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
Após salvarmos o arquivo de configuração .cfg basta executar o script Base_agent.py sem nenhum argumento:
````
python Base_agent.py
````
Outra opção para configurarmos nosso script seria executar os comandos diretamente no terminal em conjunto com a execução do script:
````
python Base_agent.py --agent_name "DQNPong30" --num_simul_frames 1000000 --lr 1e-4 --e_min 0.02 --e_lin_decay 100000 --target_update 1000 --num_states_stored 100000 --num_random_play 10000 --optimizer adam  --to_save_episodes True --random_seed 1
````
Ambas as opções de configuração irão treinar o agente com hiperparâmetros especificados pelo artigo acima com a random seed fixa (em 1) durante 1 milhão de frames como visto no resumo apresentado abaixo. **Sempre cheque o resumo apresentado pelos scripts no começo da execução para ver se todos os parâmetros estão de acordo com suas expectativas**.

<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/summary-pong30.png" height="70%" width="70%">
</p>

## Treinamento de um agente dentro do VizDoom 
Esse repositório possui em suas dependencias dois mapas para o jogo Doom, **labyrinth e labyrinth_test**, que possuem como objetivo ensinar o agente a navegação tridimensional (mais detalhes sobre esses mapas no tópico [Mapas de Doom]). Para treinar o agente na fase labyrinth utilizando a arquitetura de rede neural DRQN (proposta inicialmente em [Deep recurrent q-learning for partially observable mdps](https://arxiv.org/abs/1507.06527)[[2]](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README.md#[2])) podemos utilizar os seguintes comandos no arquivo .cfg:
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
Após salvarmos o arquivo de configuração .cfg basta executar o script Base_agent.py sem nenhum argumento:
````
python Base_agent.py
````
Outra opção para configurarmos nosso script seria executar os comandos diretamente no terminal em conjunto com a execução do script:
````
python Base_agent.py --env Doom --agent_name grayh4-LSTM --config_file_path ../DoomScenarios/labyrinth_test.cfg --network_model DRQN --is_recurrent True --optimizer adam --lr 1e-4 --num_random_play 50000 --num_states_stored 250000 --e_lin_decay 250000 --num_simul_frames 5000000 --steps_save_weights 50000 --to_save_episodes True --steps_save_episodes 100 --multi_threading True
````
O seguinte script irá treinar o agente nomeado de "grayh4-LSTM" na fase labyrinth com a arquitetura de rede neural DRQN (com camadas recorrentes do tipo LSTM) ao longo de 5 milhões de frames. Essa simulação toma vantagem do modo [multi-threading](https://github.com/Leonardo-Viana/Reinforcement-Learning#performance) para deixar a treinamento mais rápido. O resumo da simulação pode ser visto na imagem abaixo.
<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/summary-doomDRQN.png" height="70%" width="70%">
</p>

## Testando um agente treinado
O script Base_agent.py possui dois modos de execução treinamento (**train**) ou teste (**test**). O modo de treinamento é o default no qual o agente é treinado utilizando a premissa do reinforcement learning. Já no modo teste, a maioria dos hiperparâmetros de aprendizado são ignorados, o objetivo deste modo é o teste de um agente treinado. Para o script executar corretamente é necessário especificar qual a arquitetura de rede neural usada no treinamento. A seguir vemos um exemplo para o arquivo .cfg para o teste de um agente treinado com o DRQN (os pesos treinados desta simulação encontram-se neste repositório) com o jogo sendo renderizado:

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
Após salvarmos o arquivo de configuração .cfg basta executar o script Base_agent.py sem nenhum argumento:
````
python Base_agent.py
````
Outra opção para configurarmos nosso script seria executar os comandos diretamente no terminal em conjunto com a execução do script:
````
python Base_agent.py --agent_mode test --env Doom --config_file_path ../DoomScenarios/labyrinth.cfg --network_model DRQN --is_recurrent True --input_shape "84,84,1" --history_size 4 --load_weights True --weights_load_path ../Weights/Pretrained/Doom/Labyrinth/grayh4-LSTM-weights-Doom-labyrinth-5000000.h5 --agent_name doomh4-lstm-test --to_render True --to_save_states False
````

O resumo da simulação pode ser visto na imagem abaixo.
<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/summary-doomDRQN-test.png" height="70%" width="70%">
</p>

## Transfer Learning
Utilizando das opções de configuração possíveis para os scripts nesse repositório, podemos aplicar a técnica de transfer learning. Ou seja, podemos carregar um agente treinado em um ambiente e ver em quanto tempo o mesmo domina um novo ambiente. A seguir temos um exemplo de um agente treinado na fase labyrinth, aprendendo a navegar em outra fase a labyrinth_test com configurações de aprendizado bem mais rígidas.

```
agent_mode = train
env = Doom
agent_name = grayh4-LSTM
network_model = DRQN
is_recurrent = True
load_weights = True
weights_load_path = ../Weights/Weights-certos/grayh4-LSTM-weights-Doom-labyrinth-5000000.h5
config_file_path = ../DoomScenarios/labyrinth_test.cfg
optimizer = adam
lr = 1e-4
num_random_play = 0
epsilon = 0.05
num_states_stored = 50000
num_simul_frames = 500000
steps_save_weights = 50000
history_size = 4
to_save_episodes = True
steps_save_episodes = 100
multi_threading = True
```
Após salvarmos o arquivo de configuração .cfg basta executar o script Base_agent.py sem nenhum argumento:
````
python Base_agent.py
````
Outra opção para configurarmos nosso script seria executar os comandos diretamente no terminal em conjunto com a execução do script:
````
python Base_agent.py --agent_mode train --env Doom --agent_name grayh4-LSTM --network_model DRQN --is_recurrent True --load_weights True --weights_load_path ../Weights/Weights-certos/grayh4-LSTM-weights-Doom-labyrinth-5000000.h5 --config_file_path ../DoomScenarios/labyrinth_test.cfg --optimizer adam --lr 1e-4 --num_random_play 0 --epsilon 0.05 --num_states_stored 50000 --num_simul_frames 500000 ---steps_save_weights 50000 --history_size 4 --to_save_episodes True --steps_save_episodes 100 --multi_threading True 
````



