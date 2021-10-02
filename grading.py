import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

from util import *
from rules import *
from variables import * 
from ast_conversion import *

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(bert_model)

def BERTembedding(path, student=True):
    lines = read_java_codes(path)
    line_string = ' '.join(lines)
    is_syntax_errors = ASTconversion(path)[1]
    base_embeddings = model.encode([line_string])
    base_embeddings = np.mean(np.array(base_embeddings), axis=0)
    base_embeddings = base_embeddings.reshape(1, -1)
    if student:
        return base_embeddings, is_syntax_errors
    else:
        return base_embeddings

def similarity(emb1, emb2):
    manhattan_distance = cosine_similarity(emb1, emb2).squeeze()
    manhattan_vector = manhattan_distances(emb1, emb2, sum_over_features=True).squeeze()
    return manhattan_distance, manhattan_vector

def get_similarity_matrix(lecturer_code_path, student_code_path):
    lecturer_embedding = BERTembedding(lecturer_code_path, student=False)
    student_embedding, is_syntax_errors = BERTembedding(student_code_path)
    return similarity(lecturer_embedding, student_embedding), is_syntax_errors

def extract_code_paths(code_dir):
    student_code_paths = []
    for file in os.listdir(code_dir):
        if file.lower().endswith(".java") and ('student' not in file.lower()):
            lecturer_code_path = os.path.join(code_dir, file)
        elif file.lower().endswith(".java") and ('student' in file.lower()):
            student_code_path = os.path.join(code_dir, file)
            student_code_paths.append(student_code_path)
    return lecturer_code_path, student_code_paths

code_dir = 'data/6/'

lecturer_code_path, student_code_paths = extract_code_paths(code_dir)
for student_code_path in student_code_paths:
    (manhattan_distance, manhattan_vector), is_syntax_errors = get_similarity_matrix(lecturer_code_path, student_code_path)
    print('{} , {} , {}'.format(manhattan_distance, manhattan_vector, is_syntax_errors))