import tensorflow as tf 
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, ReLU, Flatten, Reshape, Lambda
from tensorflow.keras.models import Sequential, Model


import numpy as np
import matplotlib.pyplot as plt

from word_embedding import CODE2TENSOR
from variables import *

class SiameseNetwork(object):
    def __init__(self):
        self.data = CODE2TENSOR()
        X, Y, self.Errors = self.data.code_embedding()  
        
        self.input_shape = np.expand_dims(self.data.__getitem__(0)[0][:,:,0], axis=-1).shape
        
        Xlec, Xstu, Yexact, Yfunc = self.reform_data(X,Y)
        self.Xlec = Xlec
        self.Xstu = Xstu
        self.Yfunc = Yfunc
        self.Yexact = Yexact

    def normalize(self, score):
        return (score / 80.0)

    def reform_data(self, X, Y):
        Yexact = np.array([score[1] for score in Y]).squeeze()
        Yfunc = np.array([score[0] for score in Y]).squeeze()

        Yexact = self.normalize(Yexact)
        Yfunc = self.normalize(Yfunc)

        Xlec = X[:,:,:,0]
        Xstu = X[:,:,:,1]

        Xlec = np.expand_dims(Xlec, axis=-1)
        Xstu = np.expand_dims(Xstu, axis=-1)

        return Xlec, Xstu, Yexact, Yfunc

    def conv_block(self, out_channels, inputs, final_layer = False):
        if not final_layer:
            x = Conv2D(
                    out_channels, 3,
                    padding='same'
                        )(inputs)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Dropout(0.5)(x)
            x = MaxPooling2D(
                    (2,2),
                    strides=2
                        )(x)
            return x

        else:
            x = Conv2D(
                    out_channels, 3,
                    padding='same'
                        )(inputs)
            x = ReLU()(x)                        
            x = Dropout(0.5)(x)
            x = MaxPooling2D(
                    (2,2),
                    strides=2
                        )(x)
            return x

    def linear_block(self, inputs):
        x = Flatten()(inputs)
        x = Dense(512, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        return x

    def CNN(self):
        inputs = Input(shape = self.input_shape)
        conv1 = self.conv_block(16, inputs,False)
        conv2 = self.conv_block(32, conv1)
        conv3 = self.conv_block(64, conv2)
        conv4 = self.conv_block(128, conv3)
        conv5 =self.conv_block(64, conv4, True)
        linear = self.linear_block(conv5)

        model = Model(inputs, linear)
        return model
    
    @staticmethod
    def cosine_similarity(vests):
        x, y = vests
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    @staticmethod
    def cos_dist_output_shape(shapes):
        shape1, _ = shapes
        return (shape1[0],1)

    def siamese(self):
        lecturer_embedding = Input(shape = self.input_shape)
        student_embedding = Input(shape = self.input_shape)

        fv1 = self.CNN()(lecturer_embedding)
        fv2 = self.CNN()(student_embedding)

        fv1 = Reshape((1, -1))(fv1)
        fv2 = Reshape((1, -1))(fv2)

        similarity = Lambda(
                        SiameseNetwork.cosine_similarity, 
                        output_shape=SiameseNetwork.cos_dist_output_shape
                        )([fv1, fv2])

        similarity = Lambda(lambda x: K.squeeze(x, 1))(similarity)

        similarity = ReLU()(similarity)

        self.model= Model(
                    [lecturer_embedding, student_embedding], 
                    similarity
                        )
        self.model.summary()

    def train(self):
        self.model.compile(
                    loss='mse',
                    optimizer='adam'
                        )

        self.history = self.model.fit(
                                [self.Xlec, self.Xstu],
                                self.Yfunc,
                                epochs = 10,
                                batch_size = 6
                                    )
        # # self.plot_metrics()
        # self.save_model()


model = SiameseNetwork()
model.siamese()
model.train()