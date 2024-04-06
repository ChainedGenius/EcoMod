from itertools import chain

from sympy import Integral
from sympy.core.function import AppliedUndef, Derivative

from math_tasks.abstract import AnyProblem
from math_tasks.optimization import AnyOptimizationProblem, VariationalProblem, OptimalControlProblem
from utils import set_equality





def classify_task(**kwargs):
    if not kwargs.get('maximizations', []):
        return AnyProblem(**kwargs)

    if len(kwargs['maximizations']) > 1:
        raise NotImplementedError('Two optimization functionals')
    integral_sample = kwargs['maximizations'][0].find(Integral).pop()
    functions = kwargs.get('functions', [])
    if not integral_sample or (
            integral_sample and set_equality(integral_sample.find(AppliedUndef) - set(functions),
                                             integral_sample.find(AppliedUndef))):
        return AnyOptimizationProblem(**kwargs)

    # find control variables ? aka classify functions
    phase_variables = [*chain(*[[d.args[0] for d in e.find(Derivative)] for e in kwargs.get('equations', [])])]
    control_variables = [c for c in functions if c not in phase_variables]
    if not control_variables or (not phase_variables and control_variables):
        return VariationalProblem(**kwargs)

    print(phase_variables)
    print(control_variables)

    return OptimalControlProblem(**kwargs)
