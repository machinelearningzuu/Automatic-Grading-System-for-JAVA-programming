import tensorflow as tf 
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import Adam
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

    def CNN(self):
        inputs = Input(shape=(max_length,embedding_dim))
        x = Bidirectional(
                    GRU(
                       size_lstm1,
                       return_sequences=True,
                       unroll=True
                       ), name='bidirectional_lstm1')(inputs) # Bidirectional LSTM layer
        x = Bidirectional(
                    GRU(
                       size_lstm2,
                       unroll=True
                       ), name='bidirectional_lstm2')(x) # Bidirectional LSTM layer
                       
        # x = Dense(dense1, activation='relu')(x)
        x = Dense(dense1)(x) 
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        # x = Dense(dense2, activation='relu')(x) 
        x = Dense(dense2)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        # x = Dense(dense3, activation='relu')(x) 
        x = Dense(dense3, name='features')(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)
    
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
        # similarity = Lambda(lambda x: tf.keras.activations.sigmoid(x))(similarity)

        self.model= Model(
                    [lecturer_embedding, student_embedding], 
                    similarity
                        )
        self.model.summary()

    def train(self):
        self.model.compile(
                    loss=tf.keras.losses.Huber(),
                    optimizer=Adam(0.00001), 
                    metrics = ['mae']
                        )

        self.history = self.model.fit(
                                [self.Xlec, self.Xstu],
                                self.Yfunc,
                                epochs = 10,
                                batch_size = 6
                                    )
        # # self.plot_metrics()
        # self.save_model()


    def predict(self):
        return self.model.predict([self.Xlec, self.Xstu])


model = SiameseNetwork()
model.siamese()
model.train()
print(model.predict())