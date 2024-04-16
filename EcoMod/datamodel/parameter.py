from sympy import Symbol, Matrix

from EcoMod.sympy_printer import print_latex


class SymbolWrapper:
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
        self._scalar = Symbol(name, **assumptions)

    def __construct_matrix_form(self, name, m, n, **assumptions):
        values = []
        for i in range(m):
            for j in range(n):
                element_name = name + '_{' + str(i) + ',' + str(j) + '}'
                values.append(Symbol(element_name, **assumptions))
        self._matrix = Matrix(m, n, values)

    def __str__(self):
        return print_latex(self._scalar)

    # def __repr__(self):
    #     print(print_str(self))

    @classmethod
    def from_symbol(cls, symbol, **kwargs):
        obj = SymbolWrapper(symbol.name, **kwargs)
        obj._scalar = symbol
        return obj

    @classmethod
    def from_matrix(cls, matrix, *args, **kwargs):
        obj = SymbolWrapper(*args, **kwargs)
        obj._matrix = matrix
        return obj

    def reshape(self, m, n):
        self.__construct_matrix_form(self.name, m, n)



Parameter = SymbolWrapper
