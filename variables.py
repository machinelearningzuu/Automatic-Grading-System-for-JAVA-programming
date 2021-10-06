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

embeddings = 'src/word2vec_embedding.model'
scores_csv = 'scores.csv'