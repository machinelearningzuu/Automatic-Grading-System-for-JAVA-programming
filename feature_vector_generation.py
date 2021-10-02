import re
import os
import string
import gensim
import numpy as np
import pandas as pd
import scipy.stats as st
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from collections import Counter

from util import *
from variables import *
from ast_conversion import *

def generate_all_asts():
    folders = os.listdir(data_dir)
    all_asts = []
    for folder in folders:
        files = os.listdir(os.path.join(data_dir, folder))
        for file in files:
            file_path = os.path.join(data_dir, folder, file)
            ast = ASTconversion(file_path)[0]
            all_asts.append(ast)
    return all_asts

def ast_length_analysis(ast_tokens):
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
    plt.show()

def padding(ast_tokens):
    if len(ast_tokens) >= max_ength:
        pad_ast_tokens = ast_tokens[-max_ength:]
    else:
        pad_ast_tokens = [pad_token]*(max_ength-len(ast_tokens))+ ast_tokens
    return pad_ast_tokens

def tokenization_padding(asts):
    all_ast_tokens = []
    for ast in asts:
        ast_tokens = [token.strip() for token in ast.split(" ")]
        ast_tokens= padding(ast_tokens)
        all_ast_tokens.append(ast_tokens)
    return all_ast_tokens 
    
def process_asts():
    asts = generate_all_asts()
    ast_tokens = tokenization_padding(asts)
    return ast_tokens

def flat_ast_list(pad_ast_tokens):
    flat_list=[item for sublist in pad_ast_tokens for item in sublist]
    return flat_list

def pad_ast_corpus(flat_list):
    corpus=' '.join(flat_list)
    return corpus
 
def wordEmbedding():
    ast_tokens = process_asts()
    flat_list = flat_ast_list(ast_tokens)
    corpus = pad_ast_corpus(flat_list)
    model = Word2Vec(
                corpus, 
                size=100, 
                window=10,
                workers=4,
                min_count=1, 
                sg=1
                )
    return model

def my_dict(model):
    my_dict = {key:model.wv[key] for key in model.wv.vocab}
    return my_dict

model = wordEmbedding()
my_dict = my_dict(model)
print(my_dict['pad_token'])