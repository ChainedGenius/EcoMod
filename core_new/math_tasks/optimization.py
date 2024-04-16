from sympy import Integral, simplify

from math_tasks.abstract import SystemProblem
from task_classifier.validator import ExpressionDimensionValidator, CompletenessValidator


class OptimizationProblem(SystemProblem):
    def __init__(self, **kwargs):
        pass


class GlobalOptimizationProblem(OptimizationProblem):
    def __init__(self):
        pass


class DomainOptimizationProblem(OptimizationProblem):
    def __init__(self):
        pass


class VariationalProblem:
    def __init__(self, objective, eq_boundaries, config=None, diffeq_boundaries=None, ineq_boundaries=None):
        if config is None:
            config = {}
        if diffeq_boundaries is None:
            diffeq_boundaries = []
        if ineq_boundaries is None:
            ineq_boundaries = []
        self.integral = objective.find(Integral).pop()
        self.termimant = simplify(objective - self.integral)
        self.integrant = self.integral.args[0]
        self.horizon = self.integral.args[1][1:]
        self.t = self.integral.args[1][0]

        self.eq_boundaries = eq_boundaries
        self.diffeq_boundaries = diffeq_boundaries
        self.ineq_boundaries = ineq_boundaries

        self.config = config


class OptimalControlProblemValidator:
    def __new__(cls, validators=None, **kwargs):
        if validators is None:
            validators = [ExpressionDimensionValidator, CompletenessValidator]

        print('validation started')
        for v in validators:
            v(kwargs['equations'] + kwargs['inequations'] + kwargs['maximizations'], kwargs['variable_info'])

        print('validation ended')

        name = kwargs.get('name', '')
        objectives = kwargs.get('maximizations', [])
        inequations = kwargs.get('inequations', [])
        equations = kwargs.get('equations', [])
        functions = kwargs.get('functions', [])
        params = [*kwargs.get('variable_info', {}).keys()]
        dim_dict = kwargs.get('variable_info', {})
        return {
            'name': name,
            'objectives': objectives,
            'inequations': inequations,
            'equations': equations,
            'functions': functions,
            'params': params,
            'dim_dict': dim_dict
        }

