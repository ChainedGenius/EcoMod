from itertools import chain

from utils import list_substract
from errors.RWErrors import TimeVariableNotFound, ObjectiveFunctionNotFound, AnyPropertyNotFound
from deserialiser import read_model_from_tex
from sympyfier import sympify, ecomodify
from ecomod_utils import deriv_degree
from numpy import prod


class Agent(object):
    def __init__(self, objectives=None, inequations=None, equations=None, functions=None, params=None):
        if objectives is None:
            objectives = []
        if inequations is None:
            inequations = []
        if equations is None:
            equations = []
        if functions is None:
            functions = []
        if params is None:
            params = []
        self.params = params
        self.functions = functions
        self.equations = equations
        self.inequations = inequations
        self.objectives = objectives

    def __validation(self):
        # step 0: check objective
        if not self.objectives:
            raise ObjectiveFunctionNotFound()
        # step 1: check completeness
        # passed in ecomodify
        # step 2: check compliance (all functions has diff eqs, all other functions == controls)
        self.__extract_phases_controls()
        # step 3: dimension check in eq\ineq
        self.__dimension_check()
        pass

    def __dimension_check(self):
        pass

    def __extract_phases_controls(self):
        pass

    @property
    def time(self):
        # two ways: integration argument, under derivative
        from sympy import Integral
        # method 1
        try:
            ret1 = self.objectives[0].find(Integral).pop().args[1][0]
        except:
            ret1 = None
        # method 2
        try:
            ret2 = [eq for eq, deg in self.diff_degree().items() if deg == 1][0]
        except:
            ret2 = None
        if not ret1 and not ret2:
            raise TimeVariableNotFound()
        return ret1 if ret1 is not None else ret2

    @property
    def time_horizon(self):
        from sympy import Integral
        # method 1
        try:
            ret = self.objectives[0].find(Integral).pop().args[1][1:]
            return ret
        except:
            raise AnyPropertyNotFound(attr='Time Horizon')

    @property
    def transitions(self):
        # linear Differential equations in model
        return [eq for eq in self.diff_degree(1).keys()]

    @property
    def phases(self):
        from sympy import Derivative
        derivas = set(chain(*[d.find(Derivative) for d in [t for t in self.transitions]]))
        phases = [d.args[0] for d in derivas]
        indicator = prod([(p in self.functions) for p in phases])
        if indicator != 1:
            missing_var = list_substract(phases, self.functions)
            missing_transition = list_substract(self.functions, phases)
            if missing_var:
                raise AnyPropertyNotFound(attr=missing_var)
            if missing_transition:
                raise AnyPropertyNotFound(attr=missing_transition)
        return phases

    @property
    def controls(self):
        return list_substract(self.functions, self.phases)

    def diff_degree(self, deg=None):
        """
        :param deg: int. Derivative degree of returning equations
        :return:
        """
        ret = {}
        for eq in self.equations:
            ret[eq] = deriv_degree(eq)
        if deg:
            return {eq: d for eq, d in ret.items() if d == deg}
        return ret

    def read_from_tex(self, f):
        header, raw_model = read_model_from_tex(f)
        model = ecomodify(raw_model)
        return Agent(*model)

    def process(self):
        self.__validation()


if __name__ == "__main__":
    f = 'inputs/agent.tex'
    A = Agent().read_from_tex(f)
    A.process()
    print(A.equations)
    print(A.objectives)
    print(A.time)
    print(A.time_horizon)
    print(A.transitions)
    print(A.phases)
