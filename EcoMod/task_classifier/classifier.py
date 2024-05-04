import typing
from dataclasses import dataclass
from itertools import chain

from sympy import Number, Symbol, sympify, Integral
from sympy.core.function import AppliedUndef, Derivative, FunctionClass
from sympy.core.numbers import One
from sympy.core.relational import Relational

from EcoMod.utils import extract_dim_desc, set_equality, MAX, MIN
from EcoMod.math_tasks.abstract import SystemProblem
from EcoMod.math_tasks.optimization import OptimizationProblem, VariationalProblem
from EcoMod.task_classifier.validator import OptimalControlProblemValidator
from EcoMod.utils import set_equality


@dataclass
class ScalarInfo:
    # value: typing.Union[int, Number, None] = None
    # range: typing.Union[Set, Interval, ProductSet, None] = Interval(-oo, oo)
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
    def __init__(self, raw_dict, name):
        self.name = name
        self.raw_dict = raw_dict
        symbols, functions, equations, inequations, maximizations = object_classifier(raw_dict)
        self.symbols = symbols
        self.functions = functions
        self.equations = equations
        self.inequations = inequations
        self.maximizations = maximizations

        # will be filled during process and sanitize
        self.variable_info = {}  # old dimdict
        self.config_template = {}  # symbolic
        self.config = {}  # numeric ?

        self.__sanitize()
        self.__dedup()

    def __sanitize(self):
        self.__extract_external_functions()
        self.__find_function_args()
        self.__sanitize_equations()
        self.__sanitize_symbols()
        self.__extract_config()
        self.__extract_scalar_info()
        self.__sanitize_functions()
        self.config = self.config_template | self.config
        del self.config_template
        del self.raw_dict

    def __extract_external_functions(self):
        self.external_functions = [e for e in self.functions if '_' in e.name]
        self.functions = [f for f in self.functions if f not in self.external_functions]

    def __sanitize_equations(self):
        sanitized_equations = self.equations.copy()

        # remove obvious config
        for e in self.equations:
            if ((e.args[0] in self.symbols and e.args[1].is_constant())
                    or (e.args[0] in self.external_functions)):
                self.config.update({e.args[0]: e.args[1]})
                sanitized_equations.remove(e)

        # remove deep config (aka Cauchy problem conditions)
        sanitized_equations_step_1 = sanitized_equations.copy()
        for e in sanitized_equations:
            func_args = {*chain(*{i.args for i in e.find(AppliedUndef)})}
            if all([a in self.config or a.is_constant() for a in func_args]):
                sanitized_equations_step_1.remove(e)

        self.equations = sanitized_equations_step_1

    def __find_function_args(self):
        unsanitized_fargs = [*chain(*[f.args for f in self.functions])] + [
            *chain(*[[iv.args[1][0] for iv in ov.find(Integral)] for ov in self.maximizations])]
        self.fargs = [*{*[i for i in unsanitized_fargs if not i.is_Number and i not in self.config]}]

    def __sanitize_symbols(self):
        self.symbols = [s for s in self.symbols if s not in self.fargs]

    def __extract_config(self):
        for s in self.symbols + self.external_functions:
            self.config_template[s] = s

    def __extract_scalar_info(self):
        for s in self.symbols + self.fargs + self.external_functions + self.functions:
            dim, desc = extract_dim_desc(self.raw_dict[s])
            self.variable_info[s] = ScalarInfo(dim=sympify(dim) if dim else One(), desc=desc)

    def __sanitize_functions(self):
        self.functions = [f for f in self.functions if not all(i.subs(self.config).is_constant() for i in f.args)]

    def __dedup(self):
        attr_copy = self.__dict__.copy()
        self.__dict__.update({k: [*set(v)] for k, v in attr_copy.items() if isinstance(v, list)})

    @property
    def kwargs(self):
        ret = self.__dict__.copy()
        return ret


def classify_task(**kwargs):
    if not kwargs.get('maximizations', []):
        return SystemProblem(**kwargs), 1

    if len(kwargs['maximizations']) > 1:
        raise NotImplementedError('Two optimization functionals')
    integral_sample = kwargs['maximizations'][0].find(Integral).pop()
    functions = kwargs.get('functions', [])
    if not integral_sample or (
            integral_sample and set_equality(integral_sample.find(AppliedUndef) - set(functions),
                                             integral_sample.find(AppliedUndef))):
        return OptimizationProblem(**kwargs), 2

    # find control variables ? aka classify functions
    phase_variables = [*chain(*[[d.args[0] for d in e.find(Derivative)] for e in kwargs.get('equations', [])])]
    control_variables = [c for c in functions if c not in phase_variables]
    if not control_variables or (not phase_variables and control_variables):
        return VariationalProblem(**kwargs), 3

    return OptimalControlProblemValidator(**kwargs), 4
