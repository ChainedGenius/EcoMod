from sympy import Integral, simplify

from EcoMod.math_tasks.abstract import SystemProblem


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


