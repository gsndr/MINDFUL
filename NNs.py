from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, concatenate, LSTM, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, ZeroPadding2D, Activation, Add, AveragePooling2D
from keras import optimizers
from keras.models import Model
from keras import regularizers
import numpy as np
from keras.initializers import glorot_uniform

from keras.layers import LeakyReLU

np.random.seed(12)
from tensorflow import set_random_seed

set_random_seed(12)


class Models():
    def __init__(self, n_classes):
        self._nClass = n_classes



    def deepAutoEncoderUNSW(self, x_train, params):
        n_col = x_train.shape[1]
        input = Input(shape=(n_col,))

        encoded = Dense(params['first_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],
                        name='encoder2')(input)
        encoded = Dropout(params['dropout_rate'])(encoded)
        encoded = Dense(params['second_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],

                        name='encoder3')(encoded)


        decoded = Dense(params['third_layer'], activation=params['third_activation'], kernel_initializer=params['kernel_initializer'],
                        name='decoder2')(encoded)
        decoded = Dense(n_col, activation=params['third_activation'], kernel_initializer=params['kernel_initializer'],
                        name='decoder1')(decoded)


        autoencoder = Model(input=input, output=decoded)
        autoencoder.summary


        autoencoder.compile(loss=params['losses'],
                            optimizer=params['optimizer']()
                            # (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, amsgrad=False)#
                            , metrics=['acc'])

        return autoencoder


    def Conv1D(self, input_shape, params):
        input2 = Input(input_shape)

        l1 = Conv1D(params['filter'], kernel_size=1, activation=params['activation'], name='conv0', kernel_initializer=params['kernel_initializer'])(input2)
        l1 = Dropout(params['dropout_rate1'])(l1)

        l1 = Flatten()(l1)

        l1 = Dense(params['num_unit'], activation=params['first_activation'], kernel_initializer=params['kernel_initializer'])(
            l1)

        l1 = Dropout(params['dropout_rate2'])(l1)
        l1 = Dense(params['num_unit1'] , activation=params['second_activation'], kernel_initializer=params['kernel_initializer'])(l1)
        l1 = Dropout(params['dropout_rate3'])(l1)

        softmax = Dense(self._nClass, activation='softmax', kernel_initializer=params['kernel_initializer'])(l1)

        model = Model(inputs=input2, outputs=softmax)
        model.summary()
        model.compile(loss=params['losses'],
                      optimizer=params['optimizer'](),
                      metrics=['acc'])
        return model





