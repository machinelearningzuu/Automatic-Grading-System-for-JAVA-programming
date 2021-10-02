from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
from ast.JavaLexer import JavaLexer
from ast.JavaParser import JavaParser
from pprint import pformat


class AstProcessor:

    def __init__(self, logging, listener):
        self.logging = logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.listener = listener

    #★ Point 2
    def execute(self, input_source):
        parser = JavaParser(CommonTokenStream(JavaLexer(FileStream(input_source, encoding="utf-8"))))
        walker = ParseTreeWalker()
        walker.walk(self.listener, parser.compilationUnit())
        self.logger.debug('Display all data extracted by AST. \n' + pformat(self.listener.ast_info, width=160))
        return self.listener.ast_info