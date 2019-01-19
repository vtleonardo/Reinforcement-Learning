# Mapas

Juntos com os scripts que compões esse repositório, foram criados duas fases para o ambiente ViZDoom. Para a criação desses 
mapas foi pensado em um ambiente que imitasse algum problema de robótica móvel em relação a parte de navegação. O problema 
escolhido foi o de um robô móvel navegando de volta para sua plataforma de recarga de bateria. Assim, foram criados mapas
no quais o agente tem como objetivo encontrar sua plataforma de recarga de bateria o mais rápido possível. Logo, o agente é 
incentivado a não perder tempo realizando ações em um mesmo lugar e a evitar colisões de qualquer tipo com o ambiente. 


<p align="center">
 <img src="https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/images/mapa-2D.png" height="100%" width="100%">
</p>


A figura acima mostra a visão superior de ambos os mapas criados labyrinth e labyrinth test. Os círculos em vermelho sinalizam locais no qual o agente pode começar um episódio. Cada mapa é constituído de 2 salas separadas por um corredor ''curvo'' e em cada episódio o agente é colocado aleatoriamente em algum dos círculos olhando para alguma direção também aleatória. O quadro em verde (no canto superior direito no primeiro mapa e no canto inferior esquero no segundo) demonstra a posição da plataforma de recarga que o agente tem que alcançar. Embora ambos os mapas sejam praticamente iguais com a diferença da rotação, no segundo mapa o agente não pode começar em uma posição na qual ele consiga ver de imediato a plataforma de recarga devido a uma parede. Este mapa, chamado de labyrinth test foi criado para testar as capacidade de transfer learning de um agente treinado no primeiro mapa chamado de labyrinth.

<p align="center">
 <img src="https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/docs/images/mapa-3D.png" height="100%" width="100%">
</p>

A figura acima demonstra os mapas pela visão do agente. Durante o desenvolvimento dos mapas, o teto foi colocado bem acima de tal forma que não aparecesse na visão do agente. Isto foi feito para tentar aproximar a visão de um pequeno robô móvel navegando em uma sala. Além disso, cada sala possui texturas diferentes em suas paredes para que o agente possa identificar em que local ele se encontra.


