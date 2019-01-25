# Documentação
### Configurações iniciais
* [agent_mode](#agent_mode)
* [agent_name](#agent_name)
* [env](#env)
### Atari GYM exclusivo
* [include_score](#include_score)
### DOOM exclusivo
* [config_file_path](#config_file_path)
### Redes Neurais
* [network_model](#network_model)
* [normalize_input](#normalize_input)
* [is_recurrent](#is_recurrent)
### Aprendizado
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
### Configurações gerais
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
### Exclusivo do modo de teste
* [to_save_states](#to_save_states)
* [path_save_states](#path_save_states)


---
### <a name="agent_mode"></a> `agent_mode`

| Comando de Terminal | `--agent_mode <value>`    |
| :--                 | :--                       |
| **Arquivo .cfg**    | **`agent_mode = <value>`**|
| Tipo                | string                    |
| Escolhas possíveis  | train, test               |
| **Valor default**   | **train**                 |


Variável que escolhe o modo de execução do algoritmo de reinforcement learning. Existem duas opções disponíveis: train e test. 

A opção de **train** treina um agente com base no algoritmo de reinforcement learning DQN ou em suas variantes. Ou seja, o agente irá aprender a otimizar os pesos de sua rede neural com base em suas experiências vividas dentro do ambiente para maximizar sua premiação final. Logo, neste modo o algoritmo armazena as experiências passadas e otimiza a rede neural com os hiperparâmetros de aprendizagem.

A opção de **test** é usada para testar um agente que já aprendeu dentro de um ambiente. Essa opção é basicamente para o programador avaliar o desempenho do agente, gravar episódios e armazenar estados para plot.

---

### <a name="agent_name"></a> `agent_name`

| Comando de Terminal | `--agent_name <value>`    |
| :--                 | :--                       |
| **Arquivo .cfg**    | **`agent_name = <value>`**|
| Tipo                | string                    |
| **Valor default**   | **DQN**                   |


Nome do agente. Além disso, é a identificação que será utilizada para nomear os arquivos que serão salvos (Weights, Plot, Episódios, Estados).

---

### <a name="env"></a> `env`

| Comando de Terminal | `--env <value>`           |
| :--                 | :--                       |
| **Arquivo .cfg**    | **`env = <value>`**       |
| Tipo                | string                    |
| **Valor default**   | **PongNoFrameskip-v4**    |


Nome do ambiente (environment) a ser executado. Atualmente são suportados todos os jogos de atari disponíveis pela biblioteca OpenAi gym e o ambiente tridimensional ViZDoom.

Os nomes dos jogos de atari deverão seguir o seguinte template \<nome do jogo de atari\>NoFrameSkip-v4. É possível ver todos os jogos de atari disponíveis no seguinte [link](https://gym.openai.com/envs/#atari). Assim, para treinar o agente no ambiente breakout, devemos enviar para a variável env o valor BreakoutNoFrameSkip-v4 (env = BreakoutNoFrameSkip-v4 ou --env BreakoutNoFrameSkip-v4). Com a parte do "NoFrameSkip" especificamos à biblioteca que não queremos que a mesma realize o frame skipping. Desta forma temos mais controle para realizar esta etapa em nosso código (dentro do arquivo [WrapperGym.py](Environments/WrapperGym.py)).
 
Para executar o ambiente VizDoom, basta enviar para a variável env o valor doom (env = Doom ou --env Doom). 

---

### <a name="include_score"></a> `include_score`

| Comando de Terminal  | `--include_score <value>`    |
| :--                  | :--                          |
| **Arquivo .cfg**     | **`include_score = <value>`**|
| Tipo                 | bool                         |
| **Valor default**    | **False**                    |
| Exclusivo do ambiente| ATARI GYM                    |


Variável **exclusiva dos jogos de atari da biblioteca GYM** que controla se o score dos jogos de atari será incluído ou não nos frames/estados enviados pela biblioteca open ai gym. Por exemplo no jogo Pong, o score (pontuação) é localizado na parte superior da tela do jogo de atari.

---

### <a name="config_file_path"></a> `config_file_path`

| Comando de Terminal  | `--config_file_path <value>`       |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`config_file_path = <value>`**   |
| Tipo                 | string (path do sistema)           |
| **Valor default**    | **../DoomScenarios/labyrinth.cfg** |
| Exclusivo do ambiente| ViZDOOM                            |


Caminho do sistema operacional (path) para o arquivo que carrega a fase de escolha do VizDoom. A configuração da fase do VizDoom no qual o agente será treinado é feita por um arquivo .cfg, cada fase do VizDoom deverá possuir um arquivo .cfg correspondente. Portanto, para treinarmos os agentes em uma fase específica do VizDoom devemos carregar seu arquivo .cfg enviando para essa variável o seu caminho dentro do sistema operacional.

Para mais detalhes sobre os arquivos .cfg usados pela VizDoom, consulte esse [link](https://github.com/mwydmuch/ViZDoom/blob/master/doc/ConfigFile.md)

---

### <a name="network_model"></a> `network_model`

| Comando de Terminal  | `--network_model <value>`          |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`network_model = <value>`**      |
| Tipo                 | string                             |
| **Valor default**    | **DQN**                            |

Nome da função que define a arquitetura da rede neural dentro do arquivo [Networks.py](Networks.py). Para mais detalhes sobre a implementação de sua própria arquitetura de rede neural consultar o tópico: [Definindo a arquitetura da rede neural](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/nn_ptbr.md)

---

### <a name="normalize_input"></a> `normalize_input`

| Comando de Terminal  | `--normalize_input <value>`        |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`normalize_input = <value>`**    |
| Tipo                 | bool                               |
| **Valor default**    | **True**                           |

Variável que controla se é para normalizar ou não pixels de entrada da rede neural.

---

### <a name="is_recurrent"></a> `is_recurrent`

| Comando de Terminal  | `--is_recurrent <value>`        |
| :--                  | :--                             |
| **Arquivo .cfg**     | **`is_recurrent = <value>`**    |
| Tipo                 | bool                            |
| **Valor default**    | **False**                       |

Variável que diz ao script principal se a arquitetura de rede neural possui ou não camadas do tipo recorrente. Logo, se o seu modelo possuir camadas deste tipo, essa variável tem que ser enviada com o valor **True** em conjunto com a variável **network_model**, se não o script jogará uma exceção. Caso seu modelo não possua camadas do tipo recorrente, essa variável não precisa ser mandada, já que seu valor padrão é False.

---

### <a name="frame_skip"></a> `frame_skip`

| Comando de Terminal  | `--frame_skip <value>`        |
| :--                  | :--                           |
| **Arquivo .cfg**     | **`frame_skip = <value>`**    |
| Tipo                 | int                           |
| **Valor default**    | **4**                         |

Um frame válido será considerado apenas a cada \<frame_skip\> frames. Por exemplo, com um frame_skip igual a 4, somente o último frame de uma sequência de 4 frames renderizados será enviado ao código para a criação do estado. Os outros 3 frames são "descartados". Uma excelente discussão esclarecendo as ambiguidades do artigo do DQN em relação as variáveis frame_skip e [history_size](#history_size) pode ser vista [aqui](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)[[4]](https://github.com/Leonardo-Viana/Reinforcement-Learning#%5B4%5D). O termo frame em outros tópicos se refere exclusivamente aos frames válidos que são considerados pelo script.

---

### <a name="num_simul_frames"></a> `num_simul_frames`

| Comando de Terminal  | `--num_simul_frames <value>`        |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`num_simul_frames = <value>`**    |
| Tipo                 | int                                 |
| **Valor default**    | **10000000**                        |

Número de frames no qual o agente será treinado. 

---

### <a name="discount_rate"></a> `discount_rate`

| Comando de Terminal  | `--discount_rate <value>`           |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`discount_rate = <value>`**       |
| Tipo                 | float                               |
| **Valor default**    | **0.99**                            |

Fator de desconto (discount rate) gamma. 

---

### <a name="lr"></a> `lr`

| Comando de Terminal  | `--lr <value>`           |
| :--                  | :--                      |
| **Arquivo .cfg**     | **`lr = <value>`**       |
| Tipo                 | float                    |
| **Valor default**    | **0.00025**              |

Taxa de aprendizado das redes neurais. 

---

### <a name="epsilon"></a> `epsilon`

| Comando de Terminal  | `--epsilon <value>`      |
| :--                  | :--                      |
| **Arquivo .cfg**     | **`epsilon = <value>`**  |
| Tipo                 | float                    |
| **Valor default**    | **1.0**                  |

**Valor inicial** da variável épsilon da política de aprendizado e-greedy (epsilon-greedy). Essa variável balanceia o quanto de exploração de novos conhecimentos vs exploração de conhecimentos prévios o agente deve realizar. Essa variável decai ao longo da simulação. 

---

### <a name="e_min"></a> `e_min`

| Comando de Terminal  | `--e_min <value>`        |
| :--                  | :--                      |
| **Arquivo .cfg**     | **`e_min = <value>`**    |
| Tipo                 | float                    |
| **Valor default**    | **0.1**                  |

**Valor final** da variável épsilon da política de aprendizado e-greedy (épsilon-greedy) após o decaimento.

---

### <a name="decay_mode"></a> `decay_mode`

| Comando de Terminal | `--decay_mode <value>`    |
| :--                 | :--                       |
| **Arquivo .cfg**    | **`decay_mode = <value>`**|
| Tipo                | string                    |
| Escolhas possíveis  | linear, exponential       |
| **Valor default**   | **linear**                |

Variável que escolhe o tipo de decaimento da variável épsilon. Existem dois modos possíveis de decaimento da variável épsilon nesse repositório, o modo linear e exponencial.

---

### <a name="e_lin_decay"></a> `e_lin_decay`

| Comando de Terminal  | `--e_lin_decay <value>`        |
| :--                  | :--                            |
| **Arquivo .cfg**     | **`e_lin_decay = <value>`**    |
| Tipo                 | int                            |
| **Valor default**    | **1000000**                    |

Número de frames no qual o **decaimento linear** de épsilon chegará ao seu valor final. Usando os valores padrão, a variável épsilon decairá linearmente de 1.0 (100% de jogadas aleatórias) para 0.1 (10% de jogadas aleatórias) em 1 milhão de frames.

---
### <a name="e_exp_decay"></a> `e_exp_decay`

| Comando de Terminal  | `--e_exp_decay <value>`        |
| :--                  | :--                            |
| **Arquivo .cfg**     | **`e_exp_decay = <value>`**    |
| Tipo                 | int                            |
| **Valor default**    | **200000**                     |

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

| Comando de Terminal  | `--target_update <value>`      |
| :--                  | :--                            |
| **Arquivo .cfg**     | **`target_update = <value>`**  |
| Tipo                 | int                            |
| **Valor default**    | **10000**                      |

Número de frames no qual os parâmetros da Q-target network serão atualizados com os valores da Q-network.

---

### <a name="num_states_stored"></a> `num_states_stored`

| Comando de Terminal  | `--num_states_stored <value>`      |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`num_states_stored = <value>`**  |
| Tipo                 | int                                |
| **Valor default**    | **1000000**                        |

Número de experiências (estados) que serão armazenadas na replay memory.

---

### <a name="batch_size"></a> `batch_size`

| Comando de Terminal  | `--batch_size <value>`      |
| :--                  | :--                         |
| **Arquivo .cfg**     | **`batch_size = <value>`**  |
| Tipo                 | int                         |
| **Valor default**    | **32**                      |

Tamanho do batch que será utilizado para treinar as redes neurais. Em outras palavras, o número de experiências (estados) que serão amostrados da replay memory e usados para treinar a rede neural.

---

### <a name="input_shape"></a> `input_shape`

| Comando de Terminal  | `--input_shape <value>`      |
| :--                  | :--                          |
| **Arquivo .cfg**     | **`input_shape = <value>`**  |
| Tipo                 | string                       |
| **Valor default**    | **"84,84"**                  |

Dimensões nas quais os frames provindos das bibliotecas GYM/ViZDoom serão redimensionados e então amontoados para formarem as experiências/estados que serão utilizados pelo algoritmo de reinforcement learning. Os valores devem ser colocados entre aspas e com cada dimensão separada por vírgula ou espaço, seguindo o template: **"Largura, Altura, Número de canal de cores"**. Caso apenas Largura e Altura sejam enviados, é assumido que a imagem será em escala de cinza (número de canal de cores = 1). Por exemplo, para treinar o algoritmo com estados coloridos de tamanho 64 x 64 devemos enviar a essa variável o seguinte valor: "64,64,3".  

---

### <a name="history_size"></a> `history_size`

| Comando de Terminal  | `--history_size <value>`      |
| :--                  | :--                           |
| **Arquivo .cfg**     | **`history_size = <value>`**  |
| Tipo                 | int                           |
| **Valor default**    | **4**                         |

Número de frames em sequência que serão amontoados para formarem uma experiência/estado. Desta forma, o agente possuirá uma "memória", e conseguirá por exemplo saber a direção, velocidade e aceleração de objetos no ambiente. No caso da arquitetura DQN, os estados serão um volume único com formato de "Largura, Altura, Número de canal de cores * History Size". Já na arquitetura DRQN, os estados serão uma sequência de 4 volumes com formato "Largura, Altura, Número de canal de cores". Por exemplo, considere um batch de 32 amostras colhidas da replay memory, no qual cada estado é formado de frames em escala de cinza com tamanho de 84x84 pixels. A tabela a seguir mostra o formato dos tensores que serão enviados às devidas redes neurais.

|Arquitetura| Formado do Estado   |
| ---       | ---                 |
| DQN       | 32, 84, 84, 4       |
| DRQN      | 32, 4, 84, 84, 1    |

---

### <a name="num_random_play"></a> `num_random_play`

| Comando de Terminal  | `--num_random_play <value>`      |
| :--                  | :--                              |
| **Arquivo .cfg**     | **`num_random_play = <value>`**  |
| Tipo                 | int                              |
| **Valor default**    | **50000**                        |

Número de estados gerados por jogadas aleatórias feitas pelo agente antes de começar o treinamento das redes neurais com o propósito de preencher a replay memory.

---
### <a name="loss_type"></a> `loss_type`

| Comando de Terminal | `--loss_type <value>`     |
| :--                 | :--                       |
| **Arquivo .cfg**    | **`loss_type = <value>`** |
| Tipo                | string                    |
| Escolhas possíveis  | huber, MSE                |
| **Valor default**   | **huber**                 |

Tipo de loss function que será utilizada para treinar as redes neurais do agente.

---
### <a name="optimizer"></a> `optimizer`

| Comando de Terminal | `--optimizer <value>`     |
| :--                 | :--                       |
| **Arquivo .cfg**    | **`optimizer = <value>`** |
| Tipo                | string                    |
| Escolhas possíveis  | rmsprop, adam             |
| **Valor default**   | **rmsprop**               |

Tipo de optimizer que será utilizado para treinar as redes neurais do agente.

---
### <a name="load_weights"></a> `load_weights`

| Comando de Terminal  | `--load_weights <value>`        |
| :--                  | :--                             |
| **Arquivo .cfg**     | **`load_weights = <value>`**    |
| Tipo                 | bool                            |
| **Valor default**    | **False**                       |

Variável que diz ao script principal se é para carregar ou não os pesos de um rede neural de um arquivo externo .h5.

---

### <a name="weights_load_path"></a> `weights_load_path`

| Comando de Terminal  | `--weights_load_path <value>`      |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`weights_load_path = <value>`**  |
| Tipo                 | string (path do sistema)           |
| **Valor default**    | **""**                             |


Caminho do sistema operacional (path) para o arquivo .h5 que contêm os parâmetros das redes neurais a serem carregados. O valor padrão é uma string vazia. **Um parâmetro obrigatório para o TEST MODE**

---

### <a name="steps_save_weights"></a> `steps_save_weights`

| Comando de Terminal  | `--steps_save_weights <value>`      |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`steps_save_weights = <value>`**  |
| Tipo                 | int                                 |
| **Valor default**    | **50000**                           |

A cada \<steps_save_weights\> frames os pesos das redes neurais serão salvos no disco em um arquivo .h5.

---
### <a name="path_save_weights"></a> `path_save_weights`

| Comando de Terminal  | `--path_save_weights <value>`      |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`path_save_weights = <value>`**  |
| Tipo                 | string (path do sistema)           |
| **Valor default**    | **..\Weights**                     |


Caminho do sistema operacional (path) para a pasta no qual serão salvos os pesos das redes neurais em arquivos de extensão .h5.

---
### <a name="steps_save_plot"></a> `steps_save_plot`

| Comando de Terminal  | `--steps_save_plot <value>`      |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`steps_save_plot = <value>`**  |
| Tipo                 | int                                 |
| **Valor default**    | **10000**                           |

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

| Comando de Terminal  | `--path_save_plot <value>`         |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`path_save_plot = <value>`**     |
| Tipo                 | string (path do sistema)           |
| **Valor default**    | **..\Plot**                        |


Caminho do sistema operacional (path) para a pasta no qual serão salvos as variáveis a serem plotadas em arquivo .csv.

---

### <a name="to_save_episodes"></a> `to_save_episodes`

| Comando de Terminal  | `--to_save_episodes <value>`        |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`to_save_episodes = <value>`**    |
| Tipo                 | bool                                |
| **Valor default**    | **False**                           |

Variável que controla se é para salvar ou não os episódios no disco como um arquivo .gif. A seguir temos um exemplo de um episódio salvo do mapa labyrinth:

  <p align="center">
   <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/episode-Doom.gif" height="84" width="84">
  </p>
  
---
### <a name="steps_save_episodes"></a> `steps_save_episodes`

| Comando de Terminal  | `--steps_save_episodes <value>`      |
| :--                  | :--                                  |
| **Arquivo .cfg**     | **`steps_save_episodes = <value>`**  |
| Tipo                 | int                                  |
| **Valor default**    | **50**                               |

Caso o arquivo tenha que salvar os episódios ([to_save_episodes](#to_save_episodes)), eles serão salvos a cada \<steps_save_episodes\> episódios como um arquivo de imagem animada .gif.

---

### <a name="path_save_episodes"></a> `path_save_episodes`

| Comando de Terminal  | `--path_save_episodes <value>`      |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`path_save_episodes = <value>`**  |
| Tipo                 | string (path do sistema)           |
| **Valor default**    | **..\Episodes**                    |


Caminho do sistema operacional (path) para a pasta no qual serão salvos os episódios como uma imagem animada em formato .gif.

---
### <a name="silent_mode"></a> `silent_mode`

| Comando de Terminal  | `--silent_mode <value>`             |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`silent_mode = <value>`**         |
| Tipo                 | bool                                |
| **Valor default**    | **False**                           |

Caso essa variável seja verdadeira, nenhuma mensagem será exibida ao usuário.

---
### <a name="multi_gpu"></a> `multi_gpu`

| Comando de Terminal  | `--multi_gpu <value>`               |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`multi_gpu = <value>`**           |
| Tipo                 | bool                                |
| **Valor default**    | **False**                           |

Caso o usuário possua mais de uma gpu disponível e deseje usá-las para o treinamento do agente, o valor verdadeiro tem que ser atribuído a essa variável. (O gerenciamento das gpus em paralelo é feito pela biblioteca Keras)

---
### <a name="gpu_device"></a> `gpu_device`

| Comando de Terminal  | `--gpu_device <value>`               |
| :--                  | :--                                  |
| **Arquivo .cfg**     | **`gpu_device = <value>`**           |
| Tipo                 | int                                  |
| **Valor default**    | **0**                                |

Variável que permite a escolha de qual gpu a ser utilizada para o treinamento das redes neurais dos agentes. Assim, caso o usuário possua mais que uma gpu e não deseje utilizar todas elas em apenas um treinamento, é possível escolher com essa variável qual gpu utilizar, bastando atribuir o ID da gpu a essa variável e o valor False para a variável [multi_gpu](#multi_gpu). Desta forma é possível, caso haja recursos computacionais suficientes (memória, processamento), simular vários agentes simultaneamente. **Enviar o gpu_device igual -1 e a variável [multi_gpu](#multi_gpu) False fará o treinamento da rede neural rodar no processador.**

---
### <a name="multi_threading"></a> `multi_threading`

| Comando de Terminal  | `--multi_threading <value>`         |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`multi_threading = <value>`**     |
| Tipo                 | bool                                |
| **Valor default**    | **False**                           |

Se essa variável for ativada, a parte da amostragem de experiências para o treinamento da rede neural é feita paralelamente com o restante do algoritmo de aprendizagem, reduzindo, dessa forma, o tempo necessário de processamento de cada episódio. Para mais detalhes consultar o tópico [Performance](https://github.com/Leonardo-Viana/Reinforcement-Learning#performance).

---
### <a name="to_render"></a> `to_render`

| Comando de Terminal  | `--to_render <value>`               |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`to_render = <value>`**           |
| Tipo                 | bool                                |
| **Valor default**    | **False**                           |

Variável que controla se o ambiente será renderizado (mostrado na tela) para o usuário ou não, durante o treinamento ou teste. Ao renderizar o ambiente, o treinamento sofrerá uma queda enorme de processamento por episódio.

### <a name="random_seed"></a> `random_seed`

| Comando de Terminal  | `--random_seed <value>`               |
| :--                  | :--                                  |
| **Arquivo .cfg**     | **`random_seed = <value>`**           |
| Tipo                 | int                                  |
| **Valor default**    | **-1**                                |

Variável que fixa a semente dos métodos (pseudo)estocásticos. Se o valor dessa variável é -1, nenhuma semente é fixada.

---
### <a name="to_save_states"></a> `to_save_states`

| Comando de Terminal  | `--to_save_states <value>`          |
| :--                  | :--                                 |
| **Arquivo .cfg**     | **`to_save_states = <value>`**      |
| Tipo                 | bool                                |
| **Valor default**    | **False**                           |
| Exclusivo do modo    | Test                               |

Variável que controla se é para salvar ou não os estados/experiências no disco como um arquivo .gif durante o modo TEST. Os estados salvos podem ser utilizados para o plot de zonas de máxima ativação para cada camada de convolução. A seguir, temos um exemplo de um estado salvo do jogo Pong (treinado com estados coloridos):

  <p align="center">
   <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/pong-color-state.gif" height="84" width="84">
  </p>

---
### <a name="path_save_states"></a> `path_save_states`

| Comando de Terminal  | `--path_save_states <value>`       |
| :--                  | :--                                |
| **Arquivo .cfg**     | **`path_save_states = <value>`**   |
| Tipo                 | string (path do sistema)           |
| **Valor default**    | **..\States**                      |
| Exclusivo do modo    | Test                               |


Caminho do sistema operacional (path) para a pasta no qual serão salvos os estados como uma imagem animada em formato .gif.

---
