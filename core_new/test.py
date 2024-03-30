from itertools import chain
from pyclbr import Function
import typing
from sympy import Symbol, Expr, Equality, Set, Interval, ProductSet, sympify
from sympy.core.function import AppliedUndef
from sympy.core.numbers import One, Number, oo
from sympy.core.relational import Relational

from datamodel.efunction import EFunction
from datamodel.parameter import Parameter
from serializers.latex import LaTeXSerializer
from dataclasses import dataclass

from utils import extract_dim_desc


@dataclass
class ScalarInfo:
    value: typing.Union[int, Number, None] = None
    range: typing.Union[Set, Interval, ProductSet, None] = Interval(-oo, oo)
    dim: typing.Union[Symbol, Number, None] = One()
    desc: typing.Union[str, None] = ''


MAX = Symbol('MAX')
MIN = Symbol('MIN')

def object_classifier(dimdict):
    # aka ecomodify
    symbols = []
    functions = []
    equations = []
    inequations = []
    maximizations = []
    unknown = []
    for k, v in dimdict.items():
        if issubclass(type(k), Symbol):
            symbols.append(k)
        elif issubclass(type(k), AppliedUndef):
            functions.append(k)
        elif issubclass(type(k), Relational):
            if k.rel_op == '==':
                if MAX == k.args[0]:
                    maximizations.append(k.args[1])
                elif MIN == k.args[0]:
                    maximizations.append(-k.args[1])
                else:
                    equations.append(k)
            else:
                inequations.append(k)
        else:
            unknown.append(k)
    if unknown:
        raise TypeError(f'unknown objects found {unknown}')
    return symbols, functions, equations, inequations, maximizations

class TaskSanitizer:
    def __init__(self, raw_dict):
        self.raw_dict = raw_dict
        symbols, functions, equations, inequations, _ = object_classifier(raw_dict)
        self.symbols = symbols
        self.functions = functions
        self.equations = equations
        self.inequations = inequations

        # will be filled during process and sanitize
        self.scalar_info = {} #  old dimdict
        self.config_template = {} # symbolic
        self.config = {} # numeric ?

        self.__find_function_args()
        self.__sanitize_symbols()
        self.__extract_config()
        self.__sanitize_equations()
        self.__extract_scalar_info()

    def __find_function_args(self):
        self.fargs = [*chain(*[f.args for f in self.functions])]

    def __sanitize_symbols(self):
        self.symbols = [s for s in self.symbols if s not in self.fargs]

    def __extract_config(self):
        for s in self.symbols:
            self.config_template[s] = s

    def __sanitize_equations(self):
        self.config = self.config_template.copy()
        sanitized_equations = self.equations.copy()
        for e in self.equations:
            if e.args[0] in self.symbols and e.args[1].is_constant():
                self.config.update({e.args[0]: e.args[1]})
                sanitized_equations.remove(e)
        self.equations = sanitized_equations

    def __extract_scalar_info(self):
        for s in self.symbols + self.fargs:
            dim, desc = extract_dim_desc(self.raw_dict[s])
            self.scalar_info[s] = ScalarInfo(value=self.config.get(s), dim=sympify(dim), desc=desc)
        for f in self.functions:
            dim, desc = extract_dim_desc(self.raw_dict[f])
            self.scalar_info[f] = ScalarInfo(range=Interval(-oo, oo), dim=sympify(dim), desc=desc)



    def sanitize(self):
        pass




if __name__ == '__main__':
    a = LaTeXSerializer.serialize(r'C:\Users\BaTe\Documents\GitHub\EcoMod\models\inputs\agent.tex')
    a = TaskSanitizer(a)
    print(a.scalar_info)
    print(a.config_template)
    print(a.config)
    print(a.symbols)
    print(a.fargs)
    print(a.functions)
    print(a.equations)
