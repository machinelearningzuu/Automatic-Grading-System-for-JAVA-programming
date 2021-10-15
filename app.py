import os, json
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, Response, request

from variables import *

app = Flask(__name__)

from ast_conversion import ASTconversion
from rnn_grading_model import SiameseNetwork

dnn = SiameseNetwork()
dnn.run()
model = dnn.model

def extract_data():
    """
    Extracts the data from the request.
    """
    codefile = request.files['lecturer']
    filename = secure_filename(codefile.filename) 
    if filename.endswith('.jave'):
        lecturer_codes = os.listdir(lecturer_dir)
        if len(lecturer_codes) > 0:
            code_idxs = [int(code.split('.')[0]) for code in lecturer_codes]
            max_idx = max(code_idxs)
            filename = '{}.{}'.format(max_idx+1, filename)
        else:
            filename = '1.{}'.format(filename)
            
        file_dir = os.path.join(lecturer_dir,filename)
        codefile.save(file_dir)
        return file_dir
    else:
        return None

def return_ast_details():
    """
    Return the details of the AST.
    """
    file = extract_data()
    if file is not None:
        ast_details = ASTconversion(file)
        return ast_details
    else:
        return None

def lecturer_code(lecturer_codes):
    """
    Return the lecturer code from the Base
    """
    code_idxs = [int(code.split('.')[0]) for code in lecturer_codes]
    max_idx = max(code_idxs)
    filename = [code for code in lecturer_codes if code.startswith(str(max_idx))][0]
    file_dir = os.path.join(lecturer_dir,filename)
    lecturer_ast = ASTconversion(file_dir)[0]
    return lecturer_ast

@app.route("/generate_tree", methods=["POST"])
def generate_tree():
    if return_ast_details() is not None:
        ast, syntax_error, error_count = return_ast_details()
    else:
        response = {
            'status': 'Input Error',
            'message': 'Please enter a valid JAVA file'
                 }

        return Response(
                    response=json.dumps(response), 
                    status=404, 
                    mimetype="application/json"
                    )

    if syntax_error:
        response = {
            'status': 'Syntax error in the lecturer code. Can\'t proceed.',
            'message': '{}'.format(syntax_error),
            'error_count': error_count
                 }
        return Response(
                    response=json.dumps(response), 
                    status=404, 
                    mimetype="application/json"
                    )

    else:
        response = {
            'status': 'Success',
            'ast': '{}'.format(ast)
                 }
        return Response(
                    response=json.dumps(response), 
                    status=200, 
                    mimetype="application/json"
                    )

@app.route("/generate_vector", methods=["POST"])
def generate_vector():
    lecturer_codes = os.listdir(lecturer_dir)
    if len(lecturer_codes) == 0:
        response = {
            'status': 'No lecturer code found. Please upload a lecturer code.'
                 }
        return Response(
                    response=json.dumps(response), 
                    status=404, 
                    mimetype="application/json"
                    )
    
    else:
        lecturer_ast = lecturer_code(lecturer_codes)
        fv = model.get_vector(lecturer_ast)
        fv = fv.tolist() 
        response = {
            'status': 'Success',
            'feature_vector': '{}'.format(fv)
                 }
        return Response(
                response=json.dumps(response), 
                status=200, 
                mimetype="application/json"
                )

@app.route("/generate_score", methods=["POST"])
def generate_score():
    req = request.get_json(force=True)
    toggle = eval(req['toggle'])
    student_id = '{}.java'.format(req['student'])

    lecturer_codes = os.listdir(lecturer_dir)
    lecturer_ast = lecturer_code(lecturer_codes)

    student_codes = os.listdir(student_dir)
    if student_id in student_codes:
        student_code = os.path.join(student_dir, student_id)
        student_ast, syntax_error, _ =  ASTconversion(student_code)
        content_score = model.generate_score(lecturer_ast, student_ast, toggle)
        comile_score = 20 if not syntax_error else 0
        final_score = content_score + comile_score

        response = {
            'status': 'Success',
            'score': '{}'.format(final_score),
            'Syntax_errors' : 'True' if syntax_error else 'False'
                 }
        return Response(
                response=json.dumps(response), 
                status=200, 
                mimetype="application/json"
                )
    else:
        response = {
            'status': 'Student ID not found',
            'message': '{}'.format(student_id)
                 }
        return Response(
                response=json.dumps(response), 
                status=404, 
                mimetype="application/json"
                )

if __name__ == "__main__": 
    app.run(debug=True, host=host, port= port, threaded=False, use_reloader=True)