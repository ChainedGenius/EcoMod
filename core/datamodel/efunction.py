from sympy import Function, Matrix


class FunctionWrapper:
    def __init__(self, name, m=1, n=1, default_value=None, description=None, dimension=1, **assumptions):
        self.__construct_symbol_form(name, **assumptions)
        self.__construct_matrix_form(name, m, n, **assumptions)
        self.description = description
        self.dimension = dimension
        self.default_value = default_value
        self.assumptions = assumptions
        self.m = m
        self.n = n
        self.name = name

    def __construct_symbol_form(self, name, **assumptions):
        self._scalar = Function(name, **assumptions)

    def __construct_matrix_form(self, name, m, n, **assumptions):
        values = []
        for i in range(m):
            for j in range(n):
                element_name = name + '_{' + str(i) + ',' + str(j) + '}'
                values.append(Function(element_name, **assumptions)())
        self._matrix = Matrix(m, n, values)

    def __call__(self, *args, scalar=True):
        values = []
        for i in range(self.m):
            for j in range(self.n):
                element_name = self.name + '_{' + str(i) + ',' + str(j) + '}'
                values.append(Function(element_name, **self.assumptions)(*args))
        return self._scalar(*args) if scalar else Matrix(self.m, self.n, values)


EFunction = FunctionWrapper
