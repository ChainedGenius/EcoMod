from EcoMod.math_tasks.abstract import SystemProblem, NumericalProblem
from EcoMod.math_tasks.algebraic import AlgebraicSystem, AlgebraicEqSystem
from task_classifier.validator import DifferentialValidator, DifferentialAlgebraicValidator


class DifferentialSystem(NumericalProblem):
    def __init__(self, arguments, equations, config):
        super().__init__(arguments, equations, None, None, config)
        DifferentialValidator(equations)


class DiffAlgSystem(NumericalProblem):
    def __init__(self, arguments, equations, config):
        super().__init__(arguments, equations, None, None, config)
        DifferentialAlgebraicValidator(equations)
