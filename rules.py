from util import *
from variables import *

# Variable Declaration
def is_variable_declaration(line):
    line = line.strip()
    return any([line.startswith(dtype) for dtype in java_dtypes])

def extract_dtype_and_value(lines):
    dtype_dict = {}
    for idx,line in enumerate(lines):
        line = line.strip()
        if line.endswith(';'):
            line = line[:-1].strip()
        if is_variable_declaration(line):
            if not is_while_loop(lines[idx+1]):
                if '=' not in line:
                    if ',' not in line:
                        dtype, variable_name = line.split(' ')
                        variable_name = variable_name.strip()
                        dtype_dict[variable_name] = dtype.strip()
                    else:
                        dtype, variable_names = line.split(' ')
                        for variable_name in variable_names.split(','):
                            variable_name = variable_name.strip()
                            dtype_dict[variable_name] = dtype.strip()
                else:
                    dtype, variable_name, _, value = line.split(' ')

                    variable_name = variable_name.strip()
                    value = value.strip()
                    
                    dtype_dict[variable_name] = (dtype.strip(), value)
    return dtype_dict if dtype_dict else None

# Class Declaration
def is_class_declaration(line):
    line = line.strip()
    if 'class' in line:
        return line.split(' ')[1].strip() == 'class'

def extract_class_name(lines):
    class_dict = {}
    for line in lines:
        line = line.strip()
        if is_class_declaration(line):
            if line.endswith('{'):
                line = line[:-1].strip()
            class_type, _, class_name = line.split(' ')
            class_dict[class_name.strip()] = class_type.strip()
    return class_dict if class_dict else None

# Import Packages
def is_package(line):
    line = line.strip()
    return line.startswith('package')

def extract_package_name(lines):
    package_list = []
    for line in lines:
        line = line.strip()
        if is_package(line):
            if line.endswith(';'):
                line = line[:-1].strip()
            package_list.append(line.split(' ')[-1])
    return package_list if package_list else None

# Print Line
def is_print_line(line):
    line = line.strip()
    return line.startswith('System.out.println')  

def extract_print_line(lines):
    print_lines = []
    print_line_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if is_print_line(line):
            token = line.split('System.out.println')[1].strip()
            if token.startswith('(') and token.endswith(');'):
                print_lines.append(line)
            else:
                single_print = []
                single_print.append(line)
                print_line_idx = i   
        elif print_line_idx:
            single_print.append(line)
            if line.endswith(');'):
                single_print_string = ' '.join(single_print)
                print_lines.append(single_print_string)
                print_line_idx = None
    return print_lines if print_lines else None

# Extract While Loops
def is_while_loop(line):
    line = line.strip()
    return line.startswith('while')

def extract_while_loop(lines):
    while_lines = []
    while_line_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if is_while_loop(line):
            single_while = [lines[i-1]]
            single_while.append(line)
            while_line_idx = i
        elif while_line_idx:
            single_while.append(line)
            if line.endswith('}'):
                single_while_string = ' '.join(single_while)
                while_lines.append(single_while_string)
                while_line_idx = None
    return while_lines if while_lines else None

# Extract For Loops
def is_for_loop(line):
    line = line.strip()
    return line.startswith('for')

def extract_for_loop(lines):
    for_lines = []
    for_line_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if is_for_loop(line):
            for_lines.append(line)
            for_line_idx = i
        elif for_line_idx:
            for_lines.append(line)
            if line.endswith('}'):
                break
    for_string = ' '.join(for_lines)
    return for_string if for_string else None

# Extract If-else Statements
def is_if_statement(line):
    line = line.strip()
    return line.startswith('if')

def extract_if_statement(lines):
    if_statement_lines = []
    if_statement_line_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if is_if_statement(line):
            single_while = []
            single_while.append(line)
            if_statement_line_idx = i
        elif if_statement_line_idx:
            single_while.append(line)
            if line.endswith('}'):
                single_while_string = ' '.join(single_while)
                if_statement_lines.append(single_while_string)
                if_statement_line_idx = None
    return if_statement_lines if if_statement_lines else None

def retrieve_data(path, configuration):
    lines = read_java_codes(path)
    config_data = configuration(lines)
    return config_data

student_code_path = 'data/7/NumberReverseStudent2.JAVA'
print(retrieve_data(student_code_path, extract_dtype_and_value))
print(retrieve_data(student_code_path, extract_class_name)) 
print(retrieve_data(student_code_path, extract_package_name))
print(retrieve_data(student_code_path, extract_print_line))
print(retrieve_data(student_code_path, extract_while_loop))