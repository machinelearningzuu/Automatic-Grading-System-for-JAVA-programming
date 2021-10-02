import antlr4
import graphviz
from antlr4 import *
from src.ast.JavaLexer import JavaLexer
from src.ast.JavaParser import JavaParser

# lecturer_code_path = 'data/6/lecturer.JAVA'
# student_code_path = 'data/6/NumberStudent5.JAVA' 
# student_code_path = 'data/4/GradesStudent6.JAVA'

class ErrorListener(antlr4.error.ErrorListener.ErrorListener):
    
    def __init__(self):
        super(ErrorListener, self).__init__()
        self.errored_out = False

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.e = e
        self.errored_out = True
        self.recognizer = recognizer
        self.offendingSymbol = offendingSymbol
        self.error_msg = 'line {}:{} {}'.format(line, column, msg)

def ASTconversion(file_path):
    input_stream = antlr4.FileStream(file_path)

    lexer = JavaLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)

    parser = JavaParser(token_stream)

    errors = ErrorListener()
    parser.addErrorListener(errors)
    tree = parser.compilationUnit()

    is_syntax_errors = errors.errored_out
    syntax_error = errors.error_msg if is_syntax_errors else None
    error_count = parser.getNumberOfSyntaxErrors()
    return tree.toStringTree(recog=parser), syntax_error, error_count