## Exemplos
A seguir serão apresentados alguns exemplos. Todos os parâmetros podem ser passados via comandos de terminal na execução do script ou via arquivo .cfg (como visto na sessão [Utilização](https://github.com/Leonardo-Viana/Reinforcement-Learning#utiliza%C3%A7%C3%A3o)). Relembrando que os parâmetros não configurados possuem seus valores iguais ao default. Para mais informações sobre cada opção disponível e seus valores default verificar o [DOC](/docs/ptbr/doc_ptbr.md) ou utilizar o comando de terminal:
````
python Base_agent.py --help
````
### Pong treinado com DQN básico
Como primeiro exemplo treinaremos um agente utilizando os hiperparâmetros especificados pelo excelente artigo [Speeding up DQN on PyTorch: how to solve Pong in 30 minutes](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/README.md#[3]). O arquivo Base_agent.cfg deverá possuir :

```
agent_name = DQNPong30
num_simul_frames = 1000000
e_min = 0.02
e_lin_decay = 100000
target_update = 1000
num_states_stored = 100000
num_random_play = 10000
optimizer = adam
lr = 1e-4
random_seed = 1
```
E depois basta executar o script Base_agent.py sem nenhum argumento:
````
python Base_agent.py
````
Outra opção seria executar os comandos no terminal em conjunto com a execução do script:
````
python Base_agent.py --agent_name "DQNPong30" --num_simul_frames 1000000 --e_min 0.02 --e_lin_decay 100000 --target_update 1000 --num_states_stored 100000 --num_random_play 10000 --optimizer adam --lr 1e-4 --random_seed 1
````
Ambas as opções de configuração irão treinar o agente com hiperparâmetros especificados pelo artigo acima com a random seed fixa (em 1) durante 1 milhão de frames. 

### Treinamento de um agente dentro do VizDoom 
Esse repositório possui em suas dependencias dois mapas para o jogo Doom, **labyrinth e labyrinth_test**, que possuem como objetivo ensinar o agente a navegação tridimensional (mais detalhes sobre esses mapas no tópico [Mapas de Doom]). Para treinar o agente na fase labyrinth utilizando a arquitetura de rede neural DRQN proposta por [POR LINK do ARTIGO DRQN] podemos utilizar os seguintes comandos:
````
python Base_agent.py --env Doom --agent_name grayh4-LSTM --network_model DRQN --is_recurrent True --optimizer adam --lr 1e-4 --num_random_play 50000 --num_states_stored 250000 --e_lin_decay 250000 --num_simul_frames 5000000 --steps_save_weights 50000 --history_size 4 --input_shape (84,84,1) --to_save_episodes True steps_save_episodes 100 --multi_threading True
````
Ou podemos escrever dentro do arquivo Base_agent.cfg os seguintes comandos:
````
env = Doom
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
history_size = 4
input_shape = (84,84,1)
to_save_episodes = True
steps_save_episodes = 100
multi_threading = True
````
E depois basta executar o script Base_agent.py sem nenhum argumento:
````
python Base_agent.py
````
### Testando um agente treinado
O script Base_agent.py possui dois modos de execução treinamento (**train**) ou teste (**test**). O modo de treinamento é o default no qual o agente é treinado utilizando a premissa do reinforcement learning. Já no modo teste, a maioria dos hiperparâmetros de aprendizado são ignorados, o objetivo deste modo é o teste de um agente treinado. A seguir vemos um exemplo do teste de um agente treinado com o DQN (os pesos treinados desta simulação encontram-se neste repositório) com o jogo serendo renderizado:
````
python Base_agent.py --agent_mode test --env Doom --load_weights True --weights_load_path ../Weights/Pretrained/Doom/Labyrinth/grayh4-weights-Doom-labyrinth-5000000.h5 --agent_name doomh4 --to_render True --to_save_states False
````
````
agent_mode = test
env = Doom
load_weights = True
weights_load_path = ../Weights/Pretrained/Doom/Labyrinth/grayh4-weights-Doom-labyrinth-5000000.h5
agent_name = doomh4
to_render = True
to_save_states = False
````
