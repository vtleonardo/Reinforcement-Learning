# Reinforcement-Learning
## Características do código

- Modo de execução em paralelo do algoritmo de RL disponível.
- Ambientes bidimensionais ([OpenAi Gym](https://github.com/openai/gym)) e tridimensionais ([ViZDoom](https://github.com/mwydmuch/ViZDoom)) para o treinamento e teste de agentes.
- Possibilidade de inserção de outros ambientes para o treinamento de agentes.
- Configuração do treinamento/teste do agente via comandos no terminal ou via arquivos de configuração .cfg.
- Armazenamento de informações do treinamento em arquivos .csv e dos pesos das redes neurais como .h5.
- Facilidade e robustez para definir os hiperparâmetros sem a necessidade de modificar o código.
- Facilidade para a criação de arquiteturas de redes neurais sem a necessidade de modificar o código principal.
- Simulação com frames monocromáticos ou coloridos (RGB)
- Armazenamento dos episódios ao longo do treinamento e dos estados ao longo de um teste como imagens .gif.
- Plot dos mapas de ativação, zonas de máxima ativação na imagem de entrada e imagens de entrada que maximizam determinados filtros para cada uma das camadas de convolução de um modelo treinado.
- Pesos pré-treinados para os jogos Pong e para os dois mapas de ViZDoom que acompanham esse repositório.

## Performance 
Para melhorar o tempo de processamento gasto no treinamento dos agentes foi desenvolvido uma abordagem para o algoritmo de reinforcement learning rodar em paralelo. Essa abordagem consiste basicamente em amostrar as experiências da replay memory em paralelo enquanto o algoritmo de decisão é executado, assim quanto chegamos na parte de treinamento da rede neural o custo computacional da amostragem já foi executado. A seguir temos algumas imagens comparativas entre as performances em frames/segundo do modo serial (single-threading) e paralelo (multi-threading) no treinamento de agente para jogar o jogo de Atari 2600 Pong. 

<p align="center">
 <img src="docs/fps_bar.png">
</p>
*Os testes de performance foram realizado em cpu core i7 4790K e gpu nvidia geforce gtx 970*


Como podemos observar na imagem abaixo, embora a versão em paralelo introduza um "atraso" de uma amostragem, ambos os algoritmos aprenderam com sucesso a jogar o jogo Pong.

<p align="center">
 <img src="docs/pong_desemp_reward.png">
</p>

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

## Utilização
Para começar o treinamento do agente em seu ambiente de escolha basta executar o arquivo Base_agent.py com as configurações de treinamento desejadas. Essas opções podem ser passadas via comandos de terminal ou escritas no arquivo Base_agent.cfg. **Caso algum comando de terminal seja enviado, a configuração de execução do script será feita exclusivamente por eles, e os parâmetros não enviados terão seus valores atribuídos como default.** Se nenhum parâmetro for enviado via terminal, o script procurará por um arquivo de mesmo nome com extensão .cfg. Dentro deste arquivo caso encontre configurações validas às mesmas serão lidas e de forma similar a configuração via terminal, os valores não definidos serão atribuídos aos seus valores default. Se nenhuma das opções de configuração acima seja feita, o agente será treinado com seus valores default, ou seja, serão utilizados os hiperparâmetros demonstrados no artigo [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) para o treinamento de um agente no jogo Pong (PongNoFrameSkip).

Antes do começo do treinamento do agente, o script exibe um resumo das configurações e hiperparâmetros que serão utilizados em sua execução, desta forma, é possível checar se esta tudo de acordo com o planejado.

<p align="center">
 <img src="docs/summary.png">
</p>
*NN = Neural Network

## Definindo a arquitetura da rede neural
É possível definir sua propria arquitetura de rede neural para o treinamento do seu agente. Para isso basta criar dentro do arquivo Networks.py sua própria rede neural como uma função utilizando a biblioteca [Keras](https://keras.io/) **(A arquitetura pode ser criada com a functional ou sequential API)**. Dessa forma, você pode experimentar de forma rápida e sem complicações os efeitos de diferentes arquiteturas, como por exemplo, camadas recorrentes, métodos de regularização (Dropout, distância L2), normalização, batch normalization no aprendizado do agente. Após definida sua arquitetura, o nome da função deve ser enviado como um argumento via comando de terminal com o comando:

````
--network_model "<nome_da_sua_funcao>"
````
Ou escrito no arquivo Base_agent.cfg como:
````
network_model = <nome_da_sua_funcao>
````
### Requisitos
A rede neural desensolvida deve ter como entrada um tensor de dimensão **state_input_shape** e um nome igual a **name**, além da possibilidade da escolha se os pixel das entrada serão normalizados ou não pela variavel **normalize** e deve possuir como saída um tensor com formato igual **actions_num**. A função deve ter como retorno o modelo Keras implementado pela função. Os paramêtros **state_input_shape**, **name**, **actions_num** e **normalize** são enviados ao arquivo Networks.py pelo script principal, que por sua vez, espera como retorno o modelo implementado. A seguir temos um exemplo de implementação da arquitetura (com a functional API do Keras) utilizada no artigo [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) dentro de uma função chamada **DQN**:

````
def DQN_basic(state_input_shape, actions_num, name, normalize):
    input = Input(state_input_shape, name=name)
    if normalize:
        lamb = Lambda(lambda x: (2 * x - 255) / 255.0, )(input)
        conv_1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(lamb)
    else:
        conv_1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
    conv_2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv_2)
    conv_flattened = Flatten()(conv_3)
    hidden = Dense(512, activation='relu')(conv_flattened)
    output = Dense(actions_num)(hidden)
    model = Model(inputs=input, outputs=output)
    return model
````
Essa é a arquitetura padrão executada, caso nenhuma outra seja especificada na execução do agente. O arquivo [Networks.py](Networks.py)
possui outras arquiteturas de redes neurais utilizadas por mim.
## Exemplos


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
