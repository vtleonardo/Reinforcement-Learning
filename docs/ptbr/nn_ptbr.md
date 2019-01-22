## Definindo a arquitetura da rede neural
É possível definir sua própria arquitetura de rede neural para o treinamento do seu agente. Para isso basta criar dentro do arquivo Networks.py sua própria rede neural como uma função utilizando a biblioteca [Keras](https://keras.io/) **(A arquitetura pode ser criada com a functional ou sequential API)**. Dessa modo, você pode experimentar de forma rápida e sem complicações os efeitos de diferentes arquiteturas como, por exemplo, camadas recorrentes, métodos de regularização (Dropout, distância L2), normalização, batch normalization no aprendizado do agente. Após definida sua arquitetura, o nome da função deve ser enviado como um argumento via comando de terminal com o comando:

````
python Base_agent.py --network_model "<nome_da_sua_funcao>"
````
Ou escrito no arquivo Base_agent.cfg como:
````
network_model = <nome_da_sua_funcao>
````
Caso a arquitetura possua camadas do tipo recorrente é necessário atribuir o valor verdadeiro a variável **is_recurrent** na hora da execução do script principal. Desta forma caso sua arquitetura seja recorrente o comando será:

````
python Base_agent.py --network_model "<nome_da_sua_funcao_recorrente> --is_recurrent True"
````
Ou escrito no arquivo Base_agent.cfg como:
````
network_model = <nome_da_sua_funcao_recorrente>
is_recurrent = True
````

### Requisitos
A rede neural desenvolvida deve ter como entrada um tensor de dimensão **state_input_shape** e um nome igual a **name**. Além disso, deve possibilitar a escolha: se os pixels das entradas serão normalizados ou não pela variável **normalize**; e deve possuir como saída um tensor com formato igual **actions_num**. A função deve ter como retorno o modelo Keras implementado por ela. Os parâmetros **state_input_shape**, **name**, **actions_num** e **normalize** são enviados ao arquivo Networks.py pelo script principal, que por sua vez, espera como retorno o modelo implementado. A seguir, temos um exemplo de implementação da arquitetura (com a functional API do Keras) utilizada no artigo [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) dentro de uma função chamada **DQN**:

````
def DQN(state_input_shape, actions_num, name, normalize):
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
Essa é a arquitetura padrão executada caso nenhuma outra seja especificada na execução do agente. Dentro do arquivo [Networks.py](Networks.py) há outras arquiteturas de redes neurais (com camadadas recorrentes e métodos de normalização) que servem como exemplo.
