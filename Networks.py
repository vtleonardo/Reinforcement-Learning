from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Activation, Dense, Flatten
from keras.layers import LSTM, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.models import Model


def DQN_regularized(state_input_shape, actions_num, name, normalize):
    # Regularization strength
    l = 0.01
    # Defining the type of regularization (L2 distance)
    K_reg = l2(l)
    drop_rate = 0.25
    # Defining the network's input
    input = Input(state_input_shape, name=name)
    if normalize:
        # input's normalization
        lamb = Lambda(lambda x: (2 * x - 255) / 255.0, )(input)
        conv_1 = Conv2D(32, (8, 8), strides=(4, 4), kernel_regularizer=K_reg)(lamb)
    else:
        conv_1 = Conv2D(32, (8, 8), strides=(4, 4), kernel_regularizer=K_reg)(input)
    batch_n1 = BatchNormalization()(conv_1)
    act_1 = Activation("relu")(batch_n1)
    drop_1 = Dropout(rate=drop_rate)(act_1)
    conv_2 = Conv2D(64, (4, 4), strides=(2, 2), kernel_regularizer=K_reg)(drop_1)
    batch_n2 = BatchNormalization()(conv_2)
    act_2 = Activation("relu")(batch_n2)
    drop_2 = Dropout(rate=drop_rate)(act_2)
    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer=K_reg)(drop_2)
    batch_n3 = BatchNormalization()(conv_3)
    act_3 = Activation("relu")(batch_n3)
    drop_3 = Dropout(rate=drop_rate)(act_3)
    conv_flattened = Flatten()(drop_3)
    hidden = Dense(512, kernel_regularizer=K_reg)(conv_flattened)
    batch_hidden = BatchNormalization()(hidden)
    act_hidden = Activation("relu")(batch_hidden)
    drop_hidden = Dropout(rate=drop_rate)(act_hidden)
    # Defining the output
    output = Dense(actions_num)(drop_hidden)
    model = Model(inputs=input, outputs=output)
    # Returning the model
    return model


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


def DRQN(state_input_shape, actions_num, name, normalize):
    assert len(state_input_shape) == 4, "Recurrent model! Flag is_recurrent should be set to True!"
    input = Input(state_input_shape, name=name)
    if normalize:
        lamb = Lambda(lambda x: (2 * x - 255) / 255.0, )(input)
        conv_1 = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))(lamb)
    else:
        conv_1 = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))(input)
    conv_2 = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(conv_1)
    conv_3 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))(conv_2)
    conv_flattened = TimeDistributed(Flatten())(conv_3)
    hidden = LSTM(512, activation='tanh')(conv_flattened)
    output = Dense(actions_num)(hidden)
    model = Model(inputs=input, outputs=output)
    return model

