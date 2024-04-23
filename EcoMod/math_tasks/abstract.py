from types import FunctionType

from sympy import sympify, Symbol
from sympy.core.function import AppliedUndef

from EcoMod.datamodel.config import Config
from task_classifier.validator import NumericConfigValidator, CompletenessValidator


class SystemProblem:
    def __init__(self, arguments, equations, inequations, expressions):
        self.arguments = arguments
        self.equations = equations
        self.inequations = inequations
        self.expressions = expressions


class ConfigurableProblem(SystemProblem):
    def __init__(self, arguments, equations, inequations, expressions, config):
        super().__init__(arguments, equations, inequations, expressions)
        self.config = config


class NumericalProblem(ConfigurableProblem):
    def __init__(self, arguments, equations, inequations, expressions, config):
        super().__init__(arguments, equations, inequations, expressions, config)

        self.solvable = True
        try:
            NumericConfigValidator(config=config)
            CompletenessValidator(expressions=equations + inequations + expressions,
                                  variables_info=dict.fromkeys(arguments.extend(config.keys())))
        except:
            self.solvable = False

    def nsolve(self):
        if not self.solvable:
            raise TypeError('This task contains contains more variables than equations')

    def solve(self):
        ...
