# Defining your own neural network architecture
You can define your own neural network architecture to train your agent. To do this, you just need to create within Networks.py file your own neural network as a function using the Keras library (The architecture can be created with the functional or sequential API). This way, you can quickly and easily experience the effects of different architectures such as recurrent layers, regularization  methods (dropout L2 distance), normalization, batch normalization in the agent learning process. After defining its architecture, the function name must be sent as an argument via the terminal command with the command:
````
python Base_agent.py --network_model "<function_name>"
````
Or written in the file Base_agent.cfg as:
````
network_model = <function_name>
````
If the architecture has recurrent  layers, it is necessary to assign the true value to the variable is_recurrent at the time of execution of the main script. This way, if your architecture is recurrent the command will be:
````
python Base_agent.py --network_model "<function_name_recurrent> --is_recurrent True"
````
Or written in the file Base_agent.cfg as:
````
network_model = <function_name_recurrent>
is_recurrent = True
````

## Requirements

The neural network developed must have as input a tensor of **shape state_input_shap**e and a name equal to **name**. In addition, it should be possible to choose whether or not the input pixels will be normalized by the **normalize** variable; and as output, it must have a tensor with the same shape as **actions_num**. The function must have as a return the Keras model implemented by it. The **state_input_shape**, **name**, **actions_num**, and **normalize** parameters are sent to the Networks.py file by the main script, which expects to receive the implemented model. The following example shows the implementation of the architecture (with the functional API of Keras) used in the article [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) within a function called **DQN**:
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
This is the default architecture executed if no other is specified in the agent execution. Inside the [Networks.py](https://github.com/Leonardo-Viana/Reinforcement-Learning/blob/master/Networks.py) file, there are other neural network architectures (with recurrent layers and normalization methods) that serve as examples.
