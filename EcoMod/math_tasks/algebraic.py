from sympy import Derivative

from EcoMod.math_tasks.abstract import SystemProblem, NumericalProblem
from task_classifier.validator import AlgebraicValidator


class AlgebraicSystem(NumericalProblem):
    def __init__(self, variables, equations, inequations, config):
        super().__init__(variables, equations, inequations, None, config)
        AlgebraicValidator(equations + inequations)

    def solve(self):
        ...

    def nsolve(self):
        ...


class AlgebraicEqSystem(AlgebraicSystem):
    def __init__(self, variables, equations, config):
        super().__init__(variables, equations, None, config)


class LinearAlgebraicSystem(AlgebraicEqSystem):
    def __init__(self, variables, equations, config):
        super().__init__(variables, equations, config)
