import os
import tensorflow as tf 
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, BatchNormalization, Dropout, Reshape, Lambda, Embedding


import numpy as np
import matplotlib.pyplot as plt

from word_embedding import CODE2TENSOR
from resources.set import Experiments
from variables import *

class SiameseNetwork(object):
    def __init__(self, toggle=-1):
        if toggle != -1:
            self.data = CODE2TENSOR()
            self.toggle = toggle
            # X, Y, self.Errors = self.data.code_embedding()
            X, Y, self.Errors = self.data.X, self.data.Y, self.data.Errors
            Xlec, Xstu, Yexact, Yfunc = self.reform_data(X,Y)

            self.Xlec = Xlec
            self.Xstu = Xstu

            # self.Yfunc = Yfunc
            # self.Yexact = Yexact
            if self.toggle == 1:
                self.Y = Yexact
                self.model_weights = EMsiamese_weights
                self.model_architecture = EMsiamese_architecture
            elif self.toggle == 0:
                self.Y = Yfunc
                self.model_weights = FMsiamese_weights
                self.model_architecture = FMsiamese_architecture

        else:
            self.model_weights = EMsiamese_weights
            self.model_architecture = EMsiamese_architecture

    def normalize(self, score):
        return (score / 80.0)

    def reform_data(self, X, Y):
        Yexact = np.array([score[1] for score in Y]).squeeze()
        Yfunc = np.array([score[0] for score in Y]).squeeze()

        Yexact = self.normalize(Yexact)
        Yfunc = self.normalize(Yfunc)

        # Xlec = X[:,:,:,0]
        # Xstu = X[:,:,:,1]

        Xlec = np.array([x[0] for x in X])
        Xstu = np.array([x[1] for x in X])

        return Xlec, Xstu, Yexact, Yfunc

    def RNN(self):
        inputs = Input(shape=(max_length,), name='text_inputs')
        x = Embedding(
                    output_dim=emb_dim, 
                    input_dim=self.data.vocab_size, 
                    input_length=max_length, 
                    name='embedding'
                    )(inputs)
        x = Bidirectional(
                    LSTM(
                       256,
                       return_sequences=True,
                       unroll=True
                       ), name='bidirectional_lstm1')(x) # Bidirectional LSTM layer
        x = Bidirectional(
                    LSTM(
                       128,
                       unroll=True
                       ), name='bidirectional_lstm2')(x) # Bidirectional LSTM layer
                       
        # x = Dense(dense1, activation='relu')(x)
        x = Dense(256)(x) 
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(0.5)(x)

        # x = Dense(dense2, activation='relu')(x) 
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(0.5)(x)

        # x = Dense(dense3, activation='relu')(x) 
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        output = relu(x)

        model = Model(inputs, output)
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
        lecturer_embedding = Input(shape = (max_length,))
        student_embedding = Input(shape = (max_length,))

        rnn = self.RNN()
        fv1 = rnn(lecturer_embedding)
        fv2 = rnn(student_embedding)

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
                    optimizer=Adam(0.01), 
                    metrics = ['mae']
                        )

        self.history = self.model.fit(
                                [self.Xlec, self.Xstu],
                                self.Y,
                                epochs = 10,
                                batch_size = 12
                                    )

    def save_model(self): # Save trained model
        model_json = self.model.to_json()
        with open(self.model_architecture, "w") as json_file:
            json_file.write(model_json)

        self.model.save(self.model_weights)
        print("Model Saved !!!")

    def loaded_model(self, model_weights, model_architecture): # Load and compile pretrained model
        try:
            json_file = open(model_architecture, 'r')
            loaded_model_json = json_file.read()
            json_file.close()

            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(model_weights)

            self.model.compile(
                        loss=tf.keras.losses.Huber(),
                        optimizer=Adam(0.01), 
                        metrics = ['mae']
                            )
        except:
            pass
        self.model = Experiments()
        print('Model Loading !!!')

    def tokzenization(self, ast):
        tokenizer = self.data.tokenizer_save_and_load()

        ast_seq = tokenizer.texts_to_sequences([ast]) # tokenize train data
        ast_pad = pad_sequences(
                                ast_seq, 
                                maxlen=max_length, 
                                padding=padding, 
                                truncating=trunc_type
                                )
        return ast_pad
            
    def generate_score(self, lecturer_ast, student_ast, toggle):
        if toggle:
            model_weights = EMsiamese_weights
            model_architecture = EMsiamese_architecture
        else:
            model_weights = FMsiamese_weights
            model_architecture = FMsiamese_architecture
        self.loaded_model(model_weights, model_architecture)

        lecturer_ast = self.tokzenization(lecturer_ast)
        student_ast = self.tokzenization(student_ast)

        p = self.model.predict([lecturer_ast, student_ast])
        p = p.squeeze()
        p *= 80.0
        return int(p)

    def generate_vector(self, ast):
        self.loaded_model(self.model_weights, self.model_architecture)
        ast = self.tokzenization(ast)
        return self.model.predict([ast])

    def run(self):
        if not os.path.exists(self.model_weights):
            self.siamese()
            self.train()
            self.save_model()
        else:
            self.loaded_model(self.model_weights, self.model_architecture)

# '''
#  When Training Make sure to set toggle to 0/1.
# '''
# toggle = 1
# model = SiameseNetwork(toggle)
# model.run()