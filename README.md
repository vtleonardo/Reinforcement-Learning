# Reinforcement-Learning
Contém os arquivos utilizados no trabalho de conclusão de curso (TCC) **DESENVOLVIMENTO DE UM AGENTE INTELIGENTE PARA EXPLORAÇÃO AUTÔNOMA DE UM AMBIENTE TRIDIMENSIONAL VIA VISUAL REINFORCEMENT LEARNING**, apresentado no dia 28/11/2018 para a obtenção do título de engenheiro mecatrônico pelo CEFET-MG campus Divinópolis.

**Versão de rascunho disponível em:**

https://www.dropbox.com/s/oh236l071q955ig/ModeloTCC.pdf?dl=0

## Características do código

- Modo de execução em paralelo do algoritmo de RL disponível (mais detalhes na seção 3.8 do arquivo de TCC disponibilizado acima).
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
Para melhorar o tempo de processamento gasto no treinamento dos agentes foi desenvolvido uma abordagem para o algoritmo de reinforcement learning rodar em paralelo. Essa abordagem consiste basicamente em amostrar as experiências da replay memory em paralelo enquanto o algoritmo de decisão é executado, assim quanto chegamos na parte de treinamento da rede neural o custo computacional da amostragem já foi executado. Para mais detalhes consultar a seção 3.8 do arquivo de TCC. A seguir temos algumas imagens comparativas entre as performances em frames/segundo do modo serial (single-threading) e paralelo (multi-threading) no treinamento de agente para jogar o jogo de Atari 2600 Pong.

<p align="center">
 <img src="docs/fps_bar.png">
</p>
Como podemos observar na imagem abaixo, embora a versão em paralelo introduza um "atraso" de uma amostragem, ambos os algoritmos aprendem com sucesso a jogar o jogo Pong.

<p align="center">
 <img src="docs/pong_desemp_reward.png">
</p>

## Instalação


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
