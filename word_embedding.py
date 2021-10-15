import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as st
from wordcloud import WordCloud
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from util import *
from variables import *

class CODE2TENSOR(object):
    def __init__(self):
        self.max_length = max_length
        self.pad_token = pad_token
        self.scores = pd.read_csv(scores_csv)
        
        X, Y, Errors, Allasts = self.get_data()

        self.X = X
        self.Y = Y
        self.Errors = Errors
        self.asts = Allasts

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.Xemb[idx], self.Y[idx], self.Errors[idx]

    def get_data(self):
        X = []
        Y = []
        Errors = []
        Allasts = []
        questions = os.listdir(data_dir)
        for question in questions:
            question_dir = os.path.join(data_dir, question)
            codes = os.listdir(question_dir)
            lecturer_code = os.path.join(question_dir, codes[0])
            lecturer_ast = ASTconversion(lecturer_code)[0]
            student_codes = codes[1:]
            Allasts.append(lecturer_ast)
            for i, student_code in enumerate(student_codes):
                student_code = os.path.join(question_dir, student_code)
                student_ast, syntax_error, _ = ASTconversion(student_code)
                student_marks = np.logical_and(
                                    self.scores['Student'] == i+1 , 
                                    self.scores['Question'] == int(question)
                                            )
                student_marks = self.scores[student_marks]
                assert len(student_marks) ==  1, 'only one script for each student, please check'
                functional_match_marks = student_marks['Functionality Match'].values
                exact_match_marks = student_marks['Exact Match'].values
                Y.append((functional_match_marks,exact_match_marks))
                X.append((lecturer_ast, student_ast))
                Errors.append(syntax_error)
                Allasts.append(student_ast)

        X = np.array(X)
        Y = np.array(Y)
        self.create_wordcloud(Allasts)
        self.handle_data(Allasts)

        X = [(self.handle_single_ast(lecturer_ast), self.handle_single_ast(student_ast)) for (lecturer_ast, student_ast) in X]
        return X, Y, Errors, Allasts

    def ast_length_analysis(self, ast_tokens):
        if not os.path.exists(ast_length_path):
            len_x = [len(sen) for sen in ast_tokens]
            Xlen = np.array(len_x)

            q25, q75 = np.percentile(Xlen,[.25,.75])
            bin_width = 2*(q75 - q25)*len(Xlen)**(-1/3)
            bins = int(round((Xlen.max() - Xlen.min())/bin_width))

            plt.hist(Xlen, density=True, bins=bins, label="Text Lengths")
            mn, mx = plt.xlim()
            plt.xlim(mn, mx)
            kde_xs = np.linspace(mn, mx, 300)
            kde = st.gaussian_kde(Xlen)
            plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
            plt.legend(loc="upper left")
            plt.ylabel('occurance')
            plt.xlabel('lengths')
            plt.title("Histogram of AST Lengths")
            plt.savefig(ast_length_path)
            # plt.show()

    def tokenizer_save_and_load(self, tokenizer=None):
        if tokenizer:
            file_ = open(tokenizer_weights,'wb')
            pickle.dump(tokenizer, file_, protocol=pickle.HIGHEST_PROTOCOL)
            file_.close()
        else:
            assert os.path.exists(tokenizer_weights), "Tokenizer Weights doesn't exists. Please Save before load."
            file_ = open(tokenizer_weights,'rb')
            tokenizer = pickle.load(file_)
            file_.close()
            return tokenizer

    def handle_single_ast(self, ast):
        ast_seq = self.tokenizer.texts_to_sequences([ast]) # tokenize train data
        ast_pad = pad_sequences(
                                ast_seq, 
                                maxlen=max_length, 
                                padding=padding, 
                                truncating=trunc_type
                                )# Pad Train data    
        return ast_pad.squeeze()

    def create_wordcloud(self, asts):
        if not os.path.exists(wordcloud_path):
            long_string = ','.join(list(asts))
            wordcloud = WordCloud(
                                width=1600, 
                                height=800, 
                                max_words=200, 
                                background_color='white',
                                max_font_size=200
                                )
            wordcloud.generate(long_string)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("WordCloud Distribution of Student descriptions")
            plt.savefig(wordcloud_path)
            plt.show()

    def handle_data(self, Allasts):
        if (not os.path.exists(EMsiamese_weights)) and (not os.path.exists(FMsiamese_weights)):
            tokenizer = Tokenizer(
                            oov_token=oov_tok
                                ) # Create Tokenizer Object
            tokenizer.fit_on_texts(Allasts) # Fit tokenizer with train data
            self.tokenizer_save_and_load(tokenizer)

        else:
            tokenizer = self.tokenizer_save_and_load()

        AST_seq = tokenizer.texts_to_sequences(Allasts) # tokenize train data
        self.AST_pad = pad_sequences(
                                AST_seq, 
                                maxlen=max_length, 
                                padding=padding, 
                                truncating=trunc_type
                                )# Pad Train data

        self.ast_length_analysis(AST_seq)
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index) + 1

    # def padding(self, ast_tokens):
    #     if len(ast_tokens) >= self.max_length:
    #         pad_ast_tokens = ast_tokens[-self.max_length:]
    #     else:
    #         pad_ast_tokens = [self.pad_token]*(self.max_length-len(ast_tokens))+ ast_tokens
    #     return pad_ast_tokens

    # def tokenization_padding(self):
    #     all_ast_tokens = []
    #     for ast in self.asts:
    #         ast_tokens = [token.strip() for token in ast.split(" ")]
    #         ast_tokens= self.padding(ast_tokens)
    #         all_ast_tokens.append(ast_tokens)
    #     return all_ast_tokens 
 
    # def wordEmbedding(self):
    #     tokens = self.tokenization_padding()
    #     self.model = Word2Vec(
    #                 tokens, 
    #                 size=emb_dim, 
    #                 window=10,
    #                 workers=4,
    #                 min_count=1, 
    #                 sg=1
    #                 )

    # def save_embedding_model(self):
    #     self.model.save(embeddings)

    # def load_embedding_model(self):
    #     self.model = Word2Vec.load(embeddings)

    # def embedding_dict(self):
    #     dict_ = {key:self.model.wv[key] for key in self.model.wv.vocab}
    #     return dict_

    # def process_single_ast(self, ast):
    #     ast_tokens = [token.strip() for token in ast.split(" ")]
    #     ast_tokens = self.padding(ast_tokens)
    #     return ast_tokens

    # def feature_matrix_generation(self, code_or_ast, ast=True):

    #     if not ast: 
    #         ast, syntax_error, error_count = ASTconversion(code_or_ast)
    #     else:
    #         ast = code_or_ast
    #     ast_tokens = self.process_single_ast(ast)

    #     emb_matrix = np.zeros((max_length, emb_dim))
    #     for i, token in enumerate(ast_tokens):
    #         if token not in self.model.wv.vocab:
    #             zero_vector = np.zeros((emb_dim))
    #             emb_matrix[i] = zero_vector
    #         else:
    #             emb_matrix[i] = self.model.wv[token]
    #     if not ast: 
    #         return emb_matrix, syntax_error, error_count
    #     else:
    #         return emb_matrix

    # def generate_features(self, lecturer_code_or_ast, student_code_or_ast):
    #     lecturer_docs = self.feature_matrix_generation(lecturer_code_or_ast)
    #     student_docs = self.feature_matrix_generation(student_code_or_ast)
    #     return lecturer_docs, student_docs

    # def generate_train_features_all(self):
    #     # Xemb = np.empty((len(self.X), 2, max_length, emb_dim)) 
    #     Xemb = np.empty((len(self.X), max_length, emb_dim, 2))
    #     for x in self.X:
    #         lecturer_code_path = x[0]
    #         student_code_path = x[1]
    #         lecturer_docs, student_docs = self.generate_features(lecturer_code_path, student_code_path)
            
    #         # Xemb[:,0,:,:] = lecturer_docs
    #         # Xemb[:,1,:,:] = student_docs

    #         Xemb[:,:,:,0] = lecturer_docs
    #         Xemb[:,:,:,0] = student_docs

    #     # Xemb = torch.tensor(Xemb)
    #     # self.Y = torch.tensor(self.Y)

    #     Y = self.Y
    #     self.Xemb = Xemb

    #     return Xemb, Y

    # def code_embedding(self):
    #     if not os.path.exists(embeddings):
    #         self.wordEmbedding()
    #         self.save_embedding_model()
    #     else:
    #         self.load_embedding_model()
    #     Xemb, Y = self.generate_train_features_all()
    #     return Xemb, Y, self.Errors



# if __name__ == "__main__":

#     data = CODE2TENSOR()


    # data.code_embedding()   

#     lecturer_code_path = 'data/6/lecturer.JAVA'
#     student_code_path = 'data/6/NumberStudent5.JAVA' 

#     fvg = FeatureVectorGeneration()
#     fvg.process_embeddings()
#     lecturer_docs, student_docs = fvg.generate_features(lecturer_code_path, student_code_path)

#     lecturer_emb_matrix = lecturer_docs[0]
#     student_emb_matrix = student_docs[0]
#     print(lecturer_emb_matrix.shape, student_emb_matrix.shape)