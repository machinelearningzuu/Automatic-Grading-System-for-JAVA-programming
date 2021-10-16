java_dtypes = ['byte', 'short', 'int', 'long', 'float', 'double', 'boolean', 'char']
bert_model = 'bert-base-nli-mean-tokens'

line_categories = [
            'Variable Declaration',
            'Class Declaration',
            'Methods',
            'While Loop',
            'For Loop',
            'Do While Loop',
            'If Else',
            'Switch Case',
            'Comments'
                ]   

data_dir = 'data/'

emb_dim = 200
max_length = 200
pad_token = '<pad>'
oov_tok = '<oov>'
padding = 'pre'
trunc_type = 'pre'

embeddings = 'src/word2vec_embedding.model'

EMsiamese_weights = 'src/siamese_weights_exact_match.h5'
FMsiamese_weights = 'src/siamese_weights_functional_match.h5'

EMsiamese_architecture = 'src/siamese_weights_exact_match.json'
FMsiamese_architecture = 'src/siamese_weights_functional_match.json'

tokenizer_weights = 'src/tokenizer_weights.pickle'
scores_csv = 'scores.csv'

wordcloud_path = 'visualizations/wordcloud.png'
ast_length_path = 'visualizations/ast_length.png'


host = 'localhost'
port = 5000

import os
student_dir = 'logs/student/'
lecturer_dir = 'logs/lecturer/'

if not os.path.exists(student_dir):
    os.makedirs(student_dir)

if not os.path.exists(lecturer_dir):
    os.makedirs(lecturer_dir)
