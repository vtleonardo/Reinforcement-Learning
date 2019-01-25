# Documentation
### Initial configurations
* [agent_mode](#agent_mode)
* [agent_name](#agent_name)
* [env](#env)
### Atari GYM exclusive
* [include_score](#include_score)
### DOOM exclusive
* [config_file_path](#config_file_path)
### Neural Network
* [network_model](#network_model)
* [normalize_input](#normalize_input)
* [is_recurrent](#is_recurrent)
### Learning Hyperparameters
* [frame_skip](#frame_skip)
* [num_simul_frames](#num_simul_frames)
* [discount_rate](#discount_rate)
* [lr](#lr)
* [epsilon](#epsilon)
* [e_min](#e_min)
* [decay_mode](#decay_mode)
* [e_lin_decay](#e_lin_decay)
* [e_exp_decay](#e_exp_decay)
* [target_update](#target_update)
* [num_states_stored](#num_states_stored)
* [batch_size](#batch_size)
* [input_shape](#input_shape)
* [history_size](#history_size)
* [num_random_play](#num_random_play)
* [loss_type](#loss_type)
* [optimizer](#optimizer)
### General settings
* [load_weights](#load_weights)
* [weights_load_path](#weights_load_path)
* [steps_save_weights](#steps_save_weights)
* [path_save_weights](#path_save_weights)
* [steps_save_plot](#steps_save_plot)
* [path_save_plot](#path_save_plot)
* [to_save_episodes](#to_save_episodes)
* [steps_save_episodes](#steps_save_episodes)
* [path_save_episodes](#path_save_episodes)
* [silent_mode](#silent_mode)
* [multi_gpu](#multi_gpu)
* [gpu_device](#gpu_device)
* [multi_threading](#multi_threading)
* [to_render](#to_render)
* [random_seed](#random_seed)
### Test mode exclusive
* [to_save_states](#to_save_states)
* [path_save_states](#path_save_states)

---
### <a name="agent_mode"></a> `agent_mode`

| Terminal command    | `--agent_mode <value>`    |
| :--                 | :--                       |
| **CFG     file**    | **`agent_mode = <value>`**|
| Type                | string                    |
| Possible choices    | train, test               |
| **Default value**   | **train**                 |

The variable that chooses the reinforcement learning's execution mode. There are two options available: train and test.

The **train** option trains an agent using the reinforcement learning algorithms. In other words, the agent will learn how to optimize its neural network parameters based on its experiences lived inside the environment to maximize its final reward. Therefore, in this mode, the algorithm stores the past experiences and optimize the neural network with the learning hyperparameters.

The **test** option is used to test an agent that was already trained. This option is basically to evaluate visually the agent performance, record the episodes and store the states to future plots.

---

### <a name="agent_name"></a> `agent_name`

| Terminal command    | `--agent_name <value>`    |
| :--                 | :--                       |
| **CFG file**        | **`agent_name = <value>`**|
| Type                | string                    |
| **Default value**   | **DQN**                   |


Agent's name. Besides that, it is the identification that will be used to name files that will be saved by the main algorithm (Weights, Plot, Episodes, States).

---

### <a name="env"></a> `env`

| Terminal command    | `--env <value>`           |
| :--                 | :--                       |
| **CFG file**        | **`env = <value>`**       |
| Type                | string                    |
| **Default value**   | **PongNoFrameskip-v4**    |

Environment's name to be executed. Currently, this repository supports all atari games available by the OpenAI Gym and the tridimensional environments of ViZDoom.

The name of the atari games should follow the following template <Game's name>NoFrameSkip-v4. It is possible to see all atari games available in this [link](https://gym.openai.com/envs/#atari). Thus, to train an agent in the game **breakout**, we should send to the variable end the value BreakoutNoFrameSkip-v4 (env = BreakoutNoFrameSkip-v4 or --env BreakoutNoFrameSkip-v4). The part "NoFrameSkip" tells to the openAI that we don't want it to do the frame skipping. Thus, we have more control over this in the algorithm (within (WrapperGym.py)[Environments/WrapperGym.py]).

To run the ViZDoom environment, send to the variable env the value doom (env = Doom or --env Doom). 

---

### <a name="include_score"></a> `include_score`

| Terminal command     | `--include_score <value>`    |
| :--                  | :--                          |
| **CFG file**         | **`include_score = <value>`**|
| Type                 | bool                         |
| **Default value**    | **False**                    |
| Environtment exclusive| ATARI GYM                    |


**Exclusive variable for the atari games from openAi Gym**. This variable controls if the game score will be present or not in the frames sent by the GYM package. For example, in the game Pong, the score is located in the upper part of the game screen.

---

### <a name="config_file_path"></a> `config_file_path`

| Terminal command     | `--config_file_path <value>`       |
| :--                  | :--                                |
| **CFG file**         | **`config_file_path = <value>`**   |
| Type                 | string (path do sistema)           |
| **Default value**    | **../DoomScenarios/labyrinth.cfg** |
| Environtment exclusive| ViZDOOM                            |


**Exclusive variable for the ViZDoom environment**. This variable receives a system path to the file that loads the chosen map. The map configurations in ViZDoom is done by a .cfg file, each map should have its own CFG file.  Therefore, to train an agent in a specific map in ViZDoom, we should load its CFG file sending to this variable its path.

For more details about the CFG files used by ViZDoom, see this [link](https://github.com/mwydmuch/ViZDoom/blob/master/doc/ConfigFile.md)

---

### <a name="network_model"></a> `network_model`

| Terminal command     | `--network_model <value>`          |
| :--                  | :--                                |
| **CFG file**         | **`network_model = <value>`**      |
| Type                 | string                             |
| **Default value**    | **DQN**                            |

Nome da função que define a arquitetura da rede neural dentro do arquivo [Networks.py](Networks.py). Para mais detalhes sobre a implementação de sua própria arquitetura de rede neural consultar o tópico: [Definindo a arquitetura da rede neural](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/nn_ptbr.md)

---

### <a name="normalize_input"></a> `normalize_input`

| Terminal command     | `--normalize_input <value>`        |
| :--                  | :--                                |
| **CFG file**         | **`normalize_input = <value>`**    |
| Type                 | bool                               |
| **Default value**    | **True**                           |

Variável que controla se é para normalizar ou não pixels de entrada da rede neural.

---

### <a name="is_recurrent"></a> `is_recurrent`

| Terminal command     | `--is_recurrent <value>`        |
| :--                  | :--                             |
| **CFG file**         | **`is_recurrent = <value>`**    |
| Type                 | bool                            |
| **Default value**    | **False**                       |

Variável que diz ao script principal se a arquitetura de rede neural possui ou não camadas do Type recorrente. Logo, se o seu modelo possuir camadas deste Type, essa variável tem que ser enviada com o valor **True** em conjunto com a variável **network_model**, se não o script jogará uma exceção. Caso seu modelo não possua camadas do Type recorrente, essa variável não precisa ser mandada, já que seu valor padrão é False.

---

### <a name="frame_skip"></a> `frame_skip`

| Terminal command     | `--frame_skip <value>`        |
| :--                  | :--                           |
| **CFG file**         | **`frame_skip = <value>`**    |
| Type                 | int                           |
| **Default value**    | **4**                         |

Um frame válido será considerado apenas a cada \<frame_skip\> frames. Por exemplo, com um frame_skip igual a 4, somente o último frame de uma sequência de 4 frames renderizados será enviado ao código para a criação do estado. Os outros 3 frames são "descartados". Uma excelente discussão esclarecendo as ambiguidades do artigo do DQN em relação as variáveis frame_skip e [history_size](#history_size) pode ser vista [aqui](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)[[4]](https://github.com/Leonardo-Viana/Reinforcement-Learning#%5B4%5D). O termo frame em outros tópicos se refere exclusivamente aos frames válidos que são considerados pelo script.

---

### <a name="num_simul_frames"></a> `num_simul_frames`

| Terminal command     | `--num_simul_frames <value>`        |
| :--                  | :--                                 |
| **CFG file**         | **`num_simul_frames = <value>`**    |
| Type                 | int                                 |
| **Default value**    | **10000000**                        |

Número de frames no qual o agente será treinado. 

---

### <a name="discount_rate"></a> `discount_rate`

| Terminal command     | `--discount_rate <value>`           |
| :--                  | :--                                 |
| **CFG file**         | **`discount_rate = <value>`**       |
| Type                 | float                               |
| **Default value**    | **0.99**                            |

Fator de desconto (discount rate) gamma. 

---

### <a name="lr"></a> `lr`

| Terminal command     | `--lr <value>`           |
| :--                  | :--                      |
| **CFG file**         | **`lr = <value>`**       |
| Type                 | float                    |
| **Default value**    | **0.00025**              |

Taxa de aprendizado das redes neurais. 

---

### <a name="epsilon"></a> `epsilon`

| Terminal command     | `--epsilon <value>`      |
| :--                  | :--                      |
| **CFG file**         | **`epsilon = <value>`**  |
| Type                 | float                    |
| **Default value**    | **1.0**                  |

**Valor inicial** da variável épsilon da política de aprendizado e-greedy (epsilon-greedy). Essa variável balanceia o quanto de exploração de novos conhecimentos vs exploração de conhecimentos prévios o agente deve realizar. Essa variável decai ao longo da simulação. 

---

### <a name="e_min"></a> `e_min`

| Terminal command     | `--e_min <value>`        |
| :--                  | :--                      |
| **CFG file**         | **`e_min = <value>`**    |
| Type                 | float                    |
| **Default value**    | **0.1**                  |

**Valor final** da variável épsilon da política de aprendizado e-greedy (épsilon-greedy) após o decaimento.

---

### <a name="decay_mode"></a> `decay_mode`

| Terminal command    | `--decay_mode <value>`    |
| :--                 | :--                       |
| **CFG file**        | **`decay_mode = <value>`**|
| Type                | string                    |
| Possible choices    | linear, exponential       |
| **Default value**   | **linear**                |

Variável que escolhe o Type de decaimento da variável épsilon. Existem dois modos possíveis de decaimento da variável épsilon nesse repositório, o modo linear e exponencial.

---

### <a name="e_lin_decay"></a> `e_lin_decay`

| Terminal command     | `--e_lin_decay <value>`        |
| :--                  | :--                            |
| **CFG file**         | **`e_lin_decay = <value>`**    |
| Type                 | int                            |
| **Default value**    | **1000000**                    |

Número de frames no qual o **decaimento linear** de épsilon chegará ao seu valor final. Usando os valores padrão, a variável épsilon decairá linearmente de 1.0 (100% de jogadas aleatórias) para 0.1 (10% de jogadas aleatórias) em 1 milhão de frames.

---
### <a name="e_exp_decay"></a> `e_exp_decay`

| Terminal command     | `--e_exp_decay <value>`        |
| :--                  | :--                            |
| **CFG file**         | **`e_exp_decay = <value>`**    |
| Type                 | int                            |
| **Default value**    | **200000**                     |

Constante de tempo do **decaimento exponencial** de épsilon, em outras palavras, em uma constante de tempo o valor de épsilon terá decaído em 63.2% do seu valor inicial. O decaimento exponencial se dará da seguinte forma:

|Número de constantes de tempo|Decaimento do valor total|
|---                          |---                      |
|1                            |63.2%                    |
|2                            |86.5%                    |
|3                            |95%                      |
|4                            |98.2%                    |
|5                            |99.3%                    |

Assim, em aproximadamente 5 constantes de tempo, o valor de épsilon chega ao seu valor mínimo.

---

### <a name="target_update"></a> `target_update`

| Terminal command     | `--target_update <value>`      |
| :--                  | :--                            |
| **CFG file**         | **`target_update = <value>`**  |
| Type                 | int                            |
| **Default value**    | **10000**                      |

Número de frames no qual os parâmetros da Q-target network serão atualizados com os valores da Q-network.

---

### <a name="num_states_stored"></a> `num_states_stored`

| Terminal command     | `--num_states_stored <value>`      |
| :--                  | :--                                |
| **CFG file**         | **`num_states_stored = <value>`**  |
| Type                 | int                                |
| **Default value**    | **1000000**                        |

Número de experiências (estados) que serão armazenadas na replay memory.

---

### <a name="batch_size"></a> `batch_size`

| Terminal command     | `--batch_size <value>`      |
| :--                  | :--                         |
| **CFG file**         | **`batch_size = <value>`**  |
| Type                 | int                         |
| **Default value**    | **32**                      |

Tamanho do batch que será utilizado para treinar as redes neurais. Em outras palavras, o número de experiências (estados) que serão amostrados da replay memory e usados para treinar a rede neural.

---

### <a name="input_shape"></a> `input_shape`

| Terminal command     | `--input_shape <value>`      |
| :--                  | :--                          |
| **CFG file**         | **`input_shape = <value>`**  |
| Type                 | string                       |
| **Default value**    | **"84,84"**                  |

Dimensões nas quais os frames provindos das bibliotecas GYM/ViZDoom serão redimensionados e então amontoados para formarem as experiências/estados que serão utilizados pelo algoritmo de reinforcement learning. Os valores devem ser colocados entre aspas e com cada dimensão separada por vírgula ou espaço, seguindo o template: **"Largura, Altura, Número de canal de cores"**. Caso apenas Largura e Altura sejam enviados, é assumido que a imagem será em escala de cinza (número de canal de cores = 1). Por exemplo, para treinar o algoritmo com estados coloridos de tamanho 64 x 64 devemos enviar a essa variável o seguinte valor: "64,64,3".  

---

### <a name="history_size"></a> `history_size`

| Terminal command     | `--history_size <value>`      |
| :--                  | :--                           |
| **CFG file**         | **`history_size = <value>`**  |
| Type                 | int                           |
| **Default value**    | **4**                         |

Número de frames em sequência que serão amontoados para formarem uma experiência/estado. Desta forma, o agente possuirá uma "memória", e conseguirá por exemplo saber a direção, velocidade e aceleração de objetos no ambiente. No caso da arquitetura DQN, os estados serão um volume único com formato de "Largura, Altura, Número de canal de cores * History Size". Já na arquitetura DRQN, os estados serão uma sequência de 4 volumes com formato "Largura, Altura, Número de canal de cores". Por exemplo, considere um batch de 32 amostras colhidas da replay memory, no qual cada estado é formado de frames em escala de cinza com tamanho de 84x84 pixels. A tabela a seguir mostra o formato dos tensores que serão enviados às devidas redes neurais.

|Arquitetura| Formado do Estado   |
| ---       | ---                 |
| DQN       | 32, 84, 84, 4       |
| DRQN      | 32, 4, 84, 84, 1    |

---

### <a name="num_random_play"></a> `num_random_play`

| Terminal command     | `--num_random_play <value>`      |
| :--                  | :--                              |
| **CFG file**         | **`num_random_play = <value>`**  |
| Type                 | int                              |
| **Default value**    | **50000**                        |

Número de estados gerados por jogadas aleatórias feitas pelo agente antes de começar o treinamento das redes neurais com o propósito de preencher a replay memory.

---
### <a name="loss_type"></a> `loss_type`

| Terminal command    | `--loss_type <value>`     |
| :--                 | :--                       |
| **CFG file**        | **`loss_type = <value>`** |
| Type                | string                    |
| Possible choices    | huber, MSE                |
| **Default value**   | **huber**                 |

Type de loss function que será utilizada para treinar as redes neurais do agente.

---
### <a name="optimizer"></a> `optimizer`

| Terminal command    | `--optimizer <value>`     |
| :--                 | :--                       |
| **CFG file**        | **`optimizer = <value>`** |
| Type                | string                    |
| Possible choices    | rmsprop, adam             |
| **Default value**   | **rmsprop**               |

Type de optimizer que será utilizado para treinar as redes neurais do agente.

---
### <a name="load_weights"></a> `load_weights`

| Terminal command     | `--load_weights <value>`        |
| :--                  | :--                             |
| **CFG file**         | **`load_weights = <value>`**    |
| Type                 | bool                            |
| **Default value**    | **False**                       |

Variável que diz ao script principal se é para carregar ou não os pesos de um rede neural de um arquivo externo .h5.

---

### <a name="weights_load_path"></a> `weights_load_path`

| Terminal command     | `--weights_load_path <value>`      |
| :--                  | :--                                |
| **CFG file**         | **`weights_load_path = <value>`**  |
| Type                 | string (path do sistema)           |
| **Default value**    | **""**                             |


Caminho do sistema operacional (path) para o arquivo .h5 que contêm os parâmetros das redes neurais a serem carregados. O valor padrão é uma string vazia. **Um parâmetro obrigatório para o TEST MODE**

---

### <a name="steps_save_weights"></a> `steps_save_weights`

| Terminal command     | `--steps_save_weights <value>`      |
| :--                  | :--                                 |
| **CFG file**         | **`steps_save_weights = <value>`**  |
| Type                 | int                                 |
| **Default value**    | **50000**                           |

A cada \<steps_save_weights\> frames os pesos das redes neurais serão salvos no disco em um arquivo .h5.

---
### <a name="path_save_weights"></a> `path_save_weights`

| Terminal command     | `--path_save_weights <value>`      |
| :--                  | :--                                |
| **CFG file**         | **`path_save_weights = <value>`**  |
| Type                 | string (path do sistema)           |
| **Default value**    | **..\Weights**                     |


Caminho do sistema operacional (path) para a pasta no qual serão salvos os pesos das redes neurais em arquivos de extensão .h5.

---
### <a name="steps_save_plot"></a> `steps_save_plot`

| Terminal command     | `--steps_save_plot <value>`      |
| :--                  | :--                                 |
| **CFG file**         | **`steps_save_plot = <value>`**  |
| Type                 | int                                 |
| **Default value**    | **10000**                           |

A cada \<steps_save_plot\> frames as variáveis para plot armazenadas por episódio serão salvas no disco em arquivo .csv. As variáveis por episódio salvas são:

|Variáveis          |
| ---               |
| Rewards           |
| Loss              |
| Q-value médio     |
| Número de frames  |
| Tempo             |
| Frames por segundo|
| Epsilon           |

---

### <a name="path_save_plot"></a> `path_save_plot`

| Terminal command     | `--path_save_plot <value>`         |
| :--                  | :--                                |
| **CFG file**         | **`path_save_plot = <value>`**     |
| Type                 | string (path do sistema)           |
| **Default value**    | **..\Plot**                        |


Caminho do sistema operacional (path) para a pasta no qual serão salvos as variáveis a serem plotadas em arquivo .csv.

---

### <a name="to_save_episodes"></a> `to_save_episodes`

| Terminal command     | `--to_save_episodes <value>`        |
| :--                  | :--                                 |
| **CFG file**         | **`to_save_episodes = <value>`**    |
| Type                 | bool                                |
| **Default value**    | **False**                           |

Variável que controla se é para salvar ou não os episódios no disco como um arquivo .gif. A seguir temos um exemplo de um episódio salvo do mapa labyrinth:

  <p align="center">
   <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/episode-Doom.gif" height="84" width="84">
  </p>
  
---
### <a name="steps_save_episodes"></a> `steps_save_episodes`

| Terminal command     | `--steps_save_episodes <value>`      |
| :--                  | :--                                  |
| **CFG file**         | **`steps_save_episodes = <value>`**  |
| Type                 | int                                  |
| **Default value**    | **50**                               |

Caso o arquivo tenha que salvar os episódios ([to_save_episodes](#to_save_episodes)), eles serão salvos a cada \<steps_save_episodes\> episódios como um arquivo de imagem animada .gif.

---

### <a name="path_save_episodes"></a> `path_save_episodes`

| Terminal command     | `--path_save_episodes <value>`      |
| :--                  | :--                                |
| **CFG file**         | **`path_save_episodes = <value>`**  |
| Type                 | string (path do sistema)           |
| **Default value**    | **..\Episodes**                    |


Caminho do sistema operacional (path) para a pasta no qual serão salvos os episódios como uma imagem animada em formato .gif.

---
### <a name="silent_mode"></a> `silent_mode`

| Terminal command     | `--silent_mode <value>`             |
| :--                  | :--                                 |
| **CFG file**         | **`silent_mode = <value>`**         |
| Type                 | bool                                |
| **Default value**    | **False**                           |

Caso essa variável seja verdadeira, nenhuma mensagem será exibida ao usuário.

---
### <a name="multi_gpu"></a> `multi_gpu`

| Terminal command     | `--multi_gpu <value>`               |
| :--                  | :--                                 |
| **CFG file**         | **`multi_gpu = <value>`**           |
| Type                 | bool                                |
| **Default value**    | **False**                           |

Caso o usuário possua mais de uma gpu disponível e deseje usá-las para o treinamento do agente, o valor verdadeiro tem que ser atribuído a essa variável. (O gerenciamento das gpus em paralelo é feito pela biblioteca Keras)

---
### <a name="gpu_device"></a> `gpu_device`

| Terminal command     | `--gpu_device <value>`               |
| :--                  | :--                                  |
| **CFG file**         | **`gpu_device = <value>`**           |
| Type                 | int                                  |
| **Default value**    | **0**                                |

Variável que permite a escolha de qual gpu a ser utilizada para o treinamento das redes neurais dos agentes. Assim, caso o usuário possua mais que uma gpu e não deseje utilizar todas elas em apenas um treinamento, é possível escolher com essa variável qual gpu utilizar, bastando atribuir o ID da gpu a essa variável e o valor False para a variável [multi_gpu](#multi_gpu). Desta forma é possível, caso haja recursos computacionais suficientes (memória, processamento), simular vários agentes simultaneamente. **Enviar o gpu_device igual -1 e a variável [multi_gpu](#multi_gpu) False fará o treinamento da rede neural rodar no processador.**

---
### <a name="multi_threading"></a> `multi_threading`

| Terminal command     | `--multi_threading <value>`         |
| :--                  | :--                                 |
| **CFG file**         | **`multi_threading = <value>`**     |
| Type                 | bool                                |
| **Default value**    | **False**                           |

Se essa variável for ativada, a parte da amostragem de experiências para o treinamento da rede neural é feita paralelamente com o restante do algoritmo de aprendizagem, reduzindo, dessa forma, o tempo necessário de processamento de cada episódio. Para mais detalhes consultar o tópico [Performance](https://github.com/Leonardo-Viana/Reinforcement-Learning#performance).

---
### <a name="to_render"></a> `to_render`

| Terminal command     | `--to_render <value>`               |
| :--                  | :--                                 |
| **CFG file**         | **`to_render = <value>`**           |
| Type                 | bool                                |
| **Default value**    | **False**                           |

Variável que controla se o ambiente será renderizado (mostrado na tela) para o usuário ou não, durante o treinamento ou teste. Ao renderizar o ambiente, o treinamento sofrerá uma queda enorme de processamento por episódio.

### <a name="random_seed"></a> `random_seed`

| Terminal command     | `--random_seed <value>`               |
| :--                  | :--                                  |
| **CFG file**         | **`random_seed = <value>`**           |
| Type                 | int                                  |
| **Default value**    | **-1**                                |

Variável que fixa a semente dos métodos (pseudo)estocásticos. Se o valor dessa variável é -1, nenhuma semente é fixada.

---
### <a name="to_save_states"></a> `to_save_states`

| Terminal command     | `--to_save_states <value>`          |
| :--                  | :--                                 |
| **CFG file**         | **`to_save_states = <value>`**      |
| Type                 | bool                                |
| **Default value**    | **False**                           |
| Mode exclusive       | Test                               |

Variável que controla se é para salvar ou não os estados/experiências no disco como um arquivo .gif durante o modo TEST. Os estados salvos podem ser utilizados para o plot de zonas de máxima ativação para cada camada de convolução. A seguir, temos um exemplo de um estado salvo do jogo Pong (treinado com estados coloridos):

  <p align="center">
   <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/pong-color-state.gif" height="84" width="84">
  </p>

---
### <a name="path_save_states"></a> `path_save_states`

| Terminal command     | `--path_save_states <value>`       |
| :--                  | :--                                |
| **CFG file**         | **`path_save_states = <value>`**   |
| Type                 | string (path do sistema)           |
| **Default value**    | **..\States**                      |
| Mode exclusive       | Test                               |


Caminho do sistema operacional (path) para a pasta no qual serão salvos os estados como uma imagem animada em formato .gif.

---


