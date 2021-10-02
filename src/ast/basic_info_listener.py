from ast.JavaParserListener import JavaParserListener
from ast.JavaParser import JavaParser


#★ Point 3
class BasicInfoListener(JavaParserListener):

    #★ Point 4
    def __init__(self):
        self.call_methods = []
        self.ast_info = {
            'packageName': '',
            'className': '',
            'implements': [],
            'extends': '',
            'imports': [],
            'fields': [],
            'methods': []
        }

    #★ Point 5
    # Enter a parse tree produced by JavaParser#packageDeclaration.
    def enterPackageDeclaration(self, ctx:JavaParser.PackageDeclarationContext):
        self.ast_info['packageName'] = ctx.qualifiedName().getText()

    # Enter a parse tree produced by JavaParser#importDeclaration.
    def enterImportDeclaration(self, ctx:JavaParser.ImportDeclarationContext):
        import_class = ctx.qualifiedName().getText()
        self.ast_info['imports'].append(import_class)

    # Enter a parse tree produced by JavaParser#methodDeclaration.
    def enterMethodDeclaration(self, ctx:JavaParser.MethodDeclarationContext):

        print("{0} {1} {2}".format(ctx.start.line, ctx.start.column, ctx.getText()))
        self.call_methods = []

    # Exit a parse tree produced by JavaParser#methodDeclaration.
    def exitMethodDeclaration(self, ctx:JavaParser.MethodDeclarationContext):

        #★ Point 6
        c1 = ctx.getChild(0).getText()  # ---> return type
        c2 = ctx.getChild(1).getText()  # ---> method name
        # c3 = ctx.getChild(2).getText()  # ---> params
        params = self.parse_method_params_block(ctx.getChild(2))

        method_info = {
            'returnType': c1,
            'methodName': c2,
            'params': params,
            'callMethods': self.call_methods
        }
        self.ast_info['methods'].append(method_info)

    # Enter a parse tree produced by JavaParser#methodCall.
    def enterMethodCall(self, ctx:JavaParser.MethodCallContext):
        #★ Point 7
        line_number = str(ctx.start.line)
        column_number = str(ctx.start.column)
        self.call_methods.append(line_number + ' ' + column_number + ' ' + ctx.parentCtx.getText())

    # Enter a parse tree produced by JavaParser#classDeclaration.
    def enterClassDeclaration(self, ctx:JavaParser.ClassDeclarationContext):
        child_count = int(ctx.getChildCount())
        if child_count == 7:
            # class Foo extends Bar implements Hoge
            # c1 = ctx.getChild(0)  # ---> class
            c2 = ctx.getChild(1).getText()  # ---> class name
            # c3 = ctx.getChild(2)  # ---> extends
            c4 = ctx.getChild(3).getChild(0).getText()  # ---> extends class name
            # c5 = ctx.getChild(4)  # ---> implements
            # c7 = ctx.getChild(6)  # ---> method body
            self.ast_info['className'] = c2
            self.ast_info['implements'] = self.parse_implements_block(ctx.getChild(5))
            self.ast_info['extends'] = c4
        elif child_count == 5:
            # class Foo extends Bar
            # or
            # class Foo implements Hoge
            # c1 = ctx.getChild(0)  # ---> class
            c2 = ctx.getChild(1).getText()  # ---> class name
            c3 = ctx.getChild(2).getText()  # ---> extends or implements

            # c5 = ctx.getChild(4)  # ---> method body
            self.ast_info['className'] = c2
            if c3 == 'implements':
                self.ast_info['implements'] = self.parse_implements_block(ctx.getChild(3))
            elif c3 == 'extends':
                c4 = ctx.getChild(3).getChild(0).getText()  # ---> extends class name or implements class name
                self.ast_info['extends'] = c4
        elif child_count == 3:
            # class Foo
            # c1 = ctx.getChild(0)  # ---> class
            c2 = ctx.getChild(1).getText()  # ---> class name
            # c3 = ctx.getChild(2)  # ---> method body
            self.ast_info['className'] = c2

    # Enter a parse tree produced by JavaParser#fieldDeclaration.
    def enterFieldDeclaration(self, ctx:JavaParser.FieldDeclarationContext):
        field = {
            'fieldType': ctx.getChild(0).getText(),
            'fieldDefinition': ctx.getChild(1).getText()
        }
        self.ast_info['fields'].append(field)

    def parse_implements_block(self, ctx):
        implements_child_count = int(ctx.getChildCount())
        result = []
        if implements_child_count == 1:
            impl_class = ctx.getChild(0).getText()
            result.append(impl_class)
        elif implements_child_count > 1:
            for i in range(implements_child_count):
                if i % 2 == 0:
                    impl_class = ctx.getChild(i).getText()
                    result.append(impl_class)
        return result

    def parse_method_params_block(self, ctx):
        params_exist_check = int(ctx.getChildCount())
        result = []
        # () ---> 2
        # (Foo foo) ---> 3
        # (Foo foo, Bar bar) ---> 3
        # (Foo foo, Bar bar, int count) ---> 3
        if params_exist_check == 3:
            params_child_count = int(ctx.getChild(1).getChildCount())
            if params_child_count == 1:
                param_type = ctx.getChild(1).getChild(0).getChild(0).getText()
                param_name = ctx.getChild(1).getChild(0).getChild(1).getText()
                param_info = {
                    'paramType': param_type,
                    'paramName': param_name
                }
                result.append(param_info)
            elif params_child_count > 1:
                for i in range(params_child_count):
                    if i % 2 == 0:
                        param_type = ctx.getChild(1).getChild(i).getChild(0).getText()
                        param_name = ctx.getChild(1).getChild(i).getChild(1).getText()
                        param_info = {
                            'paramType': param_type,
                            'paramName': param_name
                        }
                        result.append(param_info)
        return result