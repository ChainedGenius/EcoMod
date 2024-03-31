import typing
from dataclasses import dataclass
from itertools import chain

from sympy import Number, Set, Interval, ProductSet, oo, Symbol, sympify, Integral
from sympy.core.function import AppliedUndef, Derivative
from sympy.core.numbers import One
from sympy.core.relational import Relational

from core_new.utils import extract_dim_desc, set_equality, MAX, MIN


@dataclass
class ScalarInfo:
    value: typing.Union[int, Number, None] = None
    range: typing.Union[Set, Interval, ProductSet, None] = Interval(-oo, oo)
    dim: typing.Union[Symbol, Number, None] = One()
    desc: typing.Union[str, None] = ''



def object_classifier(raw_dict):
    # aka ecomodify
    symbols = []
    functions = []
    equations = []
    inequations = []
    maximizations = []
    unknown = []
    for k, v in raw_dict.items():
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
        symbols, functions, equations, inequations, maximizations = object_classifier(raw_dict)
        self.symbols = symbols
        self.functions = functions
        self.equations = equations
        self.inequations = inequations
        self.maximizations = maximizations

        # will be filled during process and sanitize
        self.scalar_info = {}  # old dimdict
        self.config_template = {}  # symbolic
        self.config = {}  # numeric ?

        self.__sanitize()
        self.__dedup()

    def __find_function_args(self):
        unsanitized_fargs = [*chain(*[f.args for f in self.functions])] + [
            *chain(*[[iv.args[1][0] for iv in ov.find(Integral)] for ov in self.maximizations])]
        self.fargs = [i for i in unsanitized_fargs if not i.is_Number]

    def __sanitize_symbols(self):
        self.symbols = [s for s in self.symbols if s not in self.fargs]

    def __extract_config(self):
        external_functions = [e for e in self.functions if '_' in e.name]
        for s in self.symbols + external_functions:
            self.config_template[s] = s

    def __sanitize_functions(self):
        sanitized_functions = self.functions.copy()
        external_functions = [e for e in self.functions if '_' in e.name]
        for ex in external_functions:
            sanitized_functions.remove(ex)
        self.functions = sanitized_functions

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
            self.scalar_info[s] = ScalarInfo(value=self.config.get(s), dim=sympify(dim) if dim else One(), desc=desc)
        for f in self.functions:
            dim, desc = extract_dim_desc(self.raw_dict[f])
            self.scalar_info[f] = ScalarInfo(range=Interval(-oo, oo), dim=sympify(dim) if dim else One(), desc=desc)

    def __sanitize(self):
        self.__find_function_args()
        self.__sanitize_symbols()
        self.__extract_config()
        self.__sanitize_functions()
        self.__sanitize_equations()
        self.__extract_scalar_info()

    def __dedup(self):
        attr_copy = self.__dict__.copy()
        self.__dict__.update({k: [*set(v)] for k, v in attr_copy.items() if isinstance(v, list)})

    @property
    def kwargs(self):
        ret = self.__dict__.copy()
        ret.pop('raw_dict')
        return ret

