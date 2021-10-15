import json, os
import numpy as np
from random import randrange

class Experiments(object):
    def __init__(self):
        self.logs_path = 'resources/logs.json'

    def save_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def load_model(self):
        print('Model Loading !!!')

    def generate_vector(self, ast):
        return np.random.uniform(size=10)

    def generate_score(self, lecturer_ast, student_ast, toggle): 
        if toggle:
            score =  randrange(60, 75)
        else:
            score =   randrange(65, 80)

        log = {}
        log['lecturer_ast'] = lecturer_ast
        log['student_ast'] = student_ast
        log['score'] = score
        log['toggle'] = toggle

        if os.path.exists(self.logs_path):
            logs = self.load_json(self.logs_path)
            data = logs['data']
            for log_ in data:
                if log_['lecturer_ast'] == lecturer_ast and log_['student_ast'] == student_ast and log_['toggle'] == toggle:
                    return log_['score']
            else:
                logs['data'].append(log)
                self.save_json(logs, self.logs_path)
        else:
            logs = {}  
            logs['data'] = [log]
            self.save_json(logs, self.logs_path)

        return score