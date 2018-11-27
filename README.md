# Reinforcement-Learning
Contém os arquivos utilizados no trabalho de conclusão de curso (TCC) **DESENVOLVIMENTO DE UM AGENTE INTELIGENTE PARA EXPLORAÇÃO AUTÔNOMA DE UM AMBIENTE TRIDIMENSIONAL VIA VISUAL REINFORCEMENT LEARNING**, apresentado no dia 28/11/2018 para a obtenção do título de engenheiro mecatrônico pelo CEFET-MG campus Divinópolis.

**Versão de rascunho disponível em:**

https://www.dropbox.com/s/oh236l071q955ig/ModeloTCC.pdf?dl=0

## Características do código

- Modo de execução em paralelo do algoritmo de RL disponível.
- Ambientes bidimensionais ([OpenAi Gym](https://github.com/openai/gym)) e tridimensionais ([ViZDoom](https://github.com/mwydmuch/ViZDoom)) para o treinamento e teste de agentes.
- Possibilidade de inserção de outros ambientes para o treinamento de agentes.
- Execução do código via comandos no terminal ou via arquivos de configuração .cfg.
- Armazenamento de informações do treinamento em arquivos .csv e dos pesos das redes neurais como .h5.
- Facilidade e robustez para definir os hiperparâmetros sem a necessidade de modificar o código.
- Facilidade para a criação de arquiteturas de redes neurais sem a necessidade de modificar o código principal.
- Simulação com frames monocromáticos ou coloridos (RGB)
- Armazenamento dos episódios ao longo do treinamento e dos estados ao longo de um teste como imagens .gif.
- Plot dos mapas de ativação, zonas de máxima ativação na imagem de entrada e imagens de entrada que maximizam determinados filtros para cada uma das camadas de convolução de um modelo treinado.

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
