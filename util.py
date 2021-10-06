import os
from ast_conversion import ASTconversion

from variables import *

def is_comment(line):
    line = line.strip()
    return line.startswith('//')

def read_java_codes(path):
    new_lines = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split('\n')
            line = [token.strip() for token in line if token]
            if line:
                new_lines.extend(line)
    new_lines = [token.strip() for token in new_lines if token and not is_comment(token)]
    return new_lines

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