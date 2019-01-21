# Reinforcement-Learning
Neste repositório é feita a implementação dos algoritmos de deep reinforcement learning DQN e DRQN. O deep reinforcement learning é uma técnica que objetiva treinar uma inteligência artificial, chamado de agente, a completar da melhor maneira possível um determinado objetivo dentro de um ambiente normalmente estocástico possuindo como entrada os pixels da tela do ambiente. Nesta metodologia, o agente utiliza redes neurais para aprender pelo metodo de tentativa e erro qual a melhor ação a ser tomada em cada estado (imagem da tela do ambiente) para maximizar a premiação total. 


<p align="center">
 <img src="docs/images/pong-result.gif" height="50%" width="50%">
</p>

## Características do código

- [Modo de execução em paralelo do algoritmo de RL disponível](#performance).
- Ambientes bidimensionais ([OpenAi Gym](https://github.com/openai/gym)) e tridimensionais ([ViZDoom](https://github.com/mwydmuch/ViZDoom)) para o treinamento e teste de agentes.
- [Dois mapas exclusivos para o ViZDoom simulando um problema de robótica móvel](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/map_ptbr.md).
- Configuração do treinamento/teste do agente via comandos no terminal ou via arquivos de configuração .cfg (Ver as sessões de [ exemplos](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/examples_ptbr.md) e a [documentação](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/doc_ptbr.md)).
- Armazenamento de informações do treinamento em arquivos .csv ([mais detalhes aqui](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/doc_ptbr.md#steps_save_plot)) e dos pesos das redes neurais como .h5.
- Facilidade e robustez para definir os hiperparâmetros sem a necessidade de modificar o código (Ver a sessão dos [arquivos .cfg](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/cfg_ptbr.md)).
- Facilidade para a criação de arquiteturas de redes neurais sem a necessidade de modificar o código principal (Ver a sessão [Definindo a arquitetura da rede neural](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/nn_ptbr.md)).
- Simulação com frames monocromáticos ou coloridos (RGB) ([mais detalhes aqui](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/doc_ptbr.md#input_shape)).
- Armazenamento dos episódios ao longo do treinamento e dos estados ao longo de um teste como imagens .gif.
  - Exemplo de um episódio salvo do mapa labyrinth:
  <p align="center">
   <img src="docs/images/episode-Doom.gif" height="84" width="84">
  </p>
   - Exemplo de um estado salvo do jogo Pong (treinado com estados coloridos):
  <p align="center">
   <img src="docs/images/pong-color-state.gif" height="84" width="84">
  </p>
- Pesos pré-treinados para o mapa labyrinth de ViZDoom acompanhando esse repositório ([mais informações aqui](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/ptbr/info-weights_ptbr.md)).
- Plot dos mapas de ativação, zonas de máxima ativação na imagem de entrada e imagens de entrada que maximizam determinados filtros para cada uma das camadas de convolução de um modelo treinado.
<p align="center">
 <img src="https://raw.githubusercontent.com/Leonardo-Viana/Reinforcement-Learning/master/docs/images/pong-zonas.png" height="100%" width="100%">
</p>
*Clique na imagem para abrir a imagem em tamanho real.


## <a name="performance"></a>  Performance 
Um dos principais diferenciais desse repositório é a possibilidade de executar os treinamentos dos agentes em modo paralelo otimizando assim os recursos computacionais disponíveis e diminuindo assim o tempo de simulação. O modo em paralelo consiste basicamente em amostrar as experiências da replay memory em paralelo enquanto o algoritmo de decisão é executado, assim quanto chegamos na parte de treinamento da rede neural o custo computacional da amostragem já foi executado. A figura a seguir demonstra como são executadas as abordagens serial (single-threading) e paralelo (multi-threading). 
<p align="center">
 <img src="docs/images/multi-threading.png" height="100%" width="100%">
</p>
A seguir temos algumas imagens comparativas entre as performances em frames/segundo do modo serial e paralelo no treinamento de agente para jogar o jogo de Atari 2600 Pong. Como pode ser visto a versão em paralelo é em média 50% mais rápida que a versão padrão (serial).

|Processamento|Frames/segundo médio|Tempo de simulação|
| ---         | ---                | ---              |
| | |

*Os testes de performance foram realizado em cpu core i7 4790K e gpu nvidia geforce gtx 970*

## Instalação
O código foi todo escrito e testado em python 3.6 com Windows 10. Para execução do código as seguintes bibliotecas se fazem necessárias:

````
Tensorflow (cpu ou gpu)
Keras
Pandas
Imageio
OpenCV
Matplotlib
OpenAI Gym
ViZDoom
````

Para a instalação das bibliotecas acima recomenda-se criar um [ambiente virtual](https://conda.io/docs/user-guide/tasks/manage-environments.html) com o [miniconda](https://conda.io/docs/user-guide/install/index.html). Com o ambiente virtual ativado a instalação das bibliotecas com o miniconda pode ser feita com os seguintes comandos:
Versão cpu do tensorflow
````
conda install tensorflow
````
Versão gpu do tensorflow
````
conda install tensorflow-gpu
````
Para as demais bibliotecas
````
conda install keras
conda install pandas
conda install imageio
conda install opencv
conda install matplotlib
````
Para a instalação da biblioteca open ai gym em conjunto com o ambiente de ATARI 2600 no windows, utiize o seguintes comandos:
````
pip install gym
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
````
Para mais detalhes sobre a execução dos jogos de atari no windows, consultar esse [link](https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows).

Para a instalação do ViZDoom no windows consultar esse [link](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#windows_bin)

Uma vez com todas as bibliotecas instaladas e o ambiente virtual configurado, basta dar download ou clonar esse repositório e executar o arquivo Base_agent.py para o treinamento de um agente com o algoritmo de reinforcement learning DQN ou DRQN.



## Referências
Se esse código foi útil para sua pesquisa, por favor considere citar:
```
@misc{LVTeixeira,
  author = {Leonardo Viana Teixeira},
  title = {Desenvolvimento de um agente inteligente para exploração autônoma de ambientes 3D via Visual Reinforcement Learning},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/Leonardo-Viana/Reinforcement-Learning},
}
```

## Referências bibliográficas

- <a name = "[1]"></a>[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness,
Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg
Ostrovski, et al. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). Nature,
518(7540):529, 2015.
- <a name = "[2]"></a>[2] Matthew Hausknecht and Peter Stone. [Deep recurrent q-learning for partially observable mdps](https://arxiv.org/abs/1507.06527). CoRR, abs/1507.06527, 2015.
- <a name = "[3]"></a>[3] Max Lapan. Speeding up dqn on pytorch: how to solve pong in 30 minutes. 23 de
novembro 2017. Disponível em <https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55>. Acesso em: 07 de novembro de 2018.
- <a name = "[4]"></a>[4] Daniel Seita. Frame Skipping and Pre-Processing for Deep Q-Networks on Atari 2600 Games. 25 de novembro de 2016. Disponível em <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>. Acesso em: 04 de janeiro de 2019.
