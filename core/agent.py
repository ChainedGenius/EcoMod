from itertools import chain
from typing import List
from numpy import prod

from core.logger import log
from core.market import Flow
from core.utils import iterable_substract, timeit
from core.errors.RWErrors import TimeVariableNotFound, ObjectiveFunctionNotFound, AnyPropertyNotFound, \
    DimensionCheckingFailed
from core.deserialiser import read_model_from_tex
from core.sympyfier import sympify, ecomodify
from core.ecomod_utils import deriv_degree, pi_theorem, spec_funcs, generate_symbols, span, eq2func, euler_mask, \
    transversality_mask, KKT_mask


class AbstractAgent(object):
    def __init__(self, name='', objectives=None, inequations=None, equations=None, functions=None, params=None,
                 dim_dict=None):
        if dim_dict is None:
            dim_dict = []
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
        self.dim_dict = dim_dict
        self.name = name
        # dual variables k:v where k = real phase or ineq constraint, v = dual
        self.lambdas = {}  # for conjugate arbitrary constants
        self.duals = {}  # for conjugate functions
        # linking part)
        self.links = []

    @log(comment="Agent ready for validation")
    def __validation(self):
        # step 0: check objective
        if not self.objectives:
            raise ObjectiveFunctionNotFound()
        # step 1: check completeness
        # passed in ecomodify
        # step 2: check compliance (all functions has diff eqs, all other functions == controls)
        self.__extract_phases_controls()  # there are internal exception if we had problems
        # step 3: dimension check in eq\ineq
        self.__dimension_check()
        pass

    def __dimension_check(self):
        def checker(_expr, _casted_dict, _randomize_dict, _spec_stack):
            # remove spec functions
            specs = _expr.find(lambda x: x.__class__ in spec_funcs())
            _spec_stack.extend(specs)
            _expr = _expr.xreplace({s: 1 for s in specs})
            # cast all to dummies
            _expr = _expr.replace(lambda x: x.is_Function or x.is_symbol, lambda x: Symbol(x.name))

            # deal with Integrals and Derivatives
            _expr = _expr.replace(lambda x: isinstance(x, Integral), lambda x: x.args[0] * x.args[1][0])
            _expr = _expr.replace(lambda x: isinstance(x, Derivative), lambda x: x.args[0] / x.args[1][0])
            # subs
            # randomize all
            _expr = _expr.copy()
            _expr = _expr.subs(_randomize_dict)
            _expr = (_expr.lhs / _expr.rhs).subs(_casted_dict)
            return _expr.is_Number

        from sympy import Symbol, Integral, Derivative, Eq
        from numpy.random import rand
        errors = []
        spec_stack = []
        random_coef = rand(len(self.dim_dict, ))
        casted_dict = {Symbol(kv[0].name): kv[1] * random_coef[i] for i, kv in enumerate(self.dim_dict.items())}
        randomize_dict = {Symbol(kv[0].name): Symbol(kv[0].name) * random_coef[i] for i, kv in
                          enumerate(self.dim_dict.items())}
        for expr in self.expr:
            expr_copy = expr.copy()
            if expr.rhs == 0:
                from sympy import exp
                spec_stack.append(exp(expr.lhs))
                continue
            if not checker(expr, casted_dict, randomize_dict, spec_stack):
                errors.append(expr_copy)

        # spec stack check
        spec_stack = [Eq(ss.args[0], 1) for ss in spec_stack]
        for ss in spec_stack:
            if not checker(ss, casted_dict, randomize_dict, spec_stack):
                errors.append(ss)

        if errors:
            report_str = '\n'.join([i.__str__() for i in errors])
            raise DimensionCheckingFailed(expr=report_str)

    def __extract_phases_controls(self):
        p = self.phases
        if not p:
            raise AnyPropertyNotFound(attr='Phase variables')

    @log(comment='Agent ready for analysis')
    def __generate_duals(self):
        from sympy import Symbol, Function
        # step 1: create lambdas
        lambda_factor = self.objectives + self.boundaries
        lambdas_count = len(lambda_factor)
        self.lambdas = {
            (eq2func(lambda_factor[i]) if lambda_factor[i] not in self.objectives else lambda_factor[i].args[1]): v for
            i, v in enumerate(generate_symbols(tag='lambda', count=lambdas_count, cls=Symbol))}
        # step 2: create conjugates funcs (duals)
        alpha_factors = self.transitions + self.inequations
        alpha_count = len(alpha_factors)
        self.duals = {eq2func(alpha_factors[i]): d(self.time) for i, d in
                      enumerate(generate_symbols(tag='alpha', count=alpha_count, cls=Function))}

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
            from sympy import Derivative
            ret2 = [eq.find(Derivative).pop().args[1][0] for eq, deg in self.diff_degree(deg=1).items() if deg == 1][0]
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
        return [eq for eq in self.diff_degree(deg=1).keys()]

    @property
    def phases(self):
        from sympy import Derivative
        derivas = set(chain(*[d.find(Derivative) for d in [t for t in self.transitions]]))
        phases = [d.args[0] for d in derivas]
        indicator = prod([(p in self.functions) for p in phases])
        if indicator != 1:
            missing_var = iterable_substract(phases, self.functions)
            missing_transition = iterable_substract(self.functions, phases)
            if missing_var:
                raise AnyPropertyNotFound(attr=missing_var)
            if missing_transition:
                raise AnyPropertyNotFound(attr=missing_transition)
        return phases

    @property
    def controls(self):
        return iterable_substract(self.functions, self.phases)

    @property
    def expr(self):
        return self.equations + self.inequations + self.objectives

    @property
    def variables(self):
        return self.functions + self.variables

    @property
    def boundaries(self):
        # Equality types
        return [*self.diff_degree(deg=0).keys()]

    @property
    def Lagrangian(self):
        from sympy import Integral
        return span(self.duals) + span(
            {k: v for k, v in self.lambdas.items() if k in [i.args[1] for i in self.objectives]}).replace(
            lambda x: isinstance(x, Integral), lambda x: x.args[0])

    @property
    def lagrangian(self):
        return span({k: v for k, v in self.lambdas.items() if k not in [i.args[1] for i in self.objectives]})

    @property
    def kwargs(self):
        return {
            'name': self.name,
            'objectives': self.objectives,
            'inequations': self.inequations,
            'equations': self.equations,
            'functions': self.functions,
            'params': self.params,
            'dim_dict': self.dim_dict
        }

    # ECOMOD CORE SOFT
    def euler_equations(self):
        ret = []
        for x in self.phases:
            ret.append(euler_mask(self.Lagrangian, x, self.time))
        return ret

    def transversality_conditions(self):
        ret = []
        for x in self.phases:
            # for all t_0, t_1
            ret.extend(transversality_mask(self.Lagrangian, x, self.time, self.lagrangian, *self.time_horizon))

        return ret

    def control_optimality(self):
        return [self.Lagrangian.diff(c) for c in self.controls]

    def KKT(self):
        _duals = {k: v for k, v in self.duals.items() if k not in [eq2func(e) for e in self.transitions]}
        return KKT_mask(_duals)

    def diff_degree(self, deg=None):
        """
        :param deg: int. Derivative degree of returning equations
        :return:
        """
        ret = {}
        for eq in self.equations:
            ret[eq] = deriv_degree(eq)
        # casting values to int
        ret = {k: int(v.__str__()) for k, v in ret.items()}
        if deg is not None:
            ret = [eq for eq, d in ret.items() if d == deg]
            return {k: deg for k in ret}
        return ret

    @classmethod
    @log(comment='\nParsing new agent file.')
    def read_from_tex(cls, f):
        from pathlib import Path
        name = Path(f).stem
        header, raw_model = read_model_from_tex(f)
        model = ecomodify(raw_model)
        return cls(name, *model)

    @log(comment='Agent ready for economic processing')
    def process(self, skip_validation=False):
        if not skip_validation:
            self.__validation()

        self.__generate_duals()


class LinkedAgent(AbstractAgent):
    def __init__(self, *args):
        super().__init__(*args)
        self.flows = []

    def add_flow(self, flow: Flow):
        if flow.receiver != self and flow.producer != self:
            print('This flow do not affect this agent')
        else:
            self.flows.append(flow)

    def delete_flow(self, flow: Flow):
        try:
            self.flows.remove(flow)
        except ValueError:
            # TODO: create custom error
            print('There is no such flow in Agent file')

    def print_flows(self):
        return "\n".join([f'[{flow}]' for flow in self.flows])

    @staticmethod
    def from_abstract(a: AbstractAgent):
        kwargs = a.kwargs
        return LinkedAgent(*kwargs.values())


class LAgentValidator(object):
    # for links mb redirected to Lagents class
    def __variable_check(self, agents: List[AbstractAgent]):
        pass

    def __dimension_check(self, agents: List[AbstractAgent]):
        pass

    # isolated model checks
    def __variable_completeness(self, agents: List[AbstractAgent]):
        pass

    def validate_agents(self, agents: List[AbstractAgent]):
        self.__variable_check(agents)
        self.__dimension_check(agents)
        self.__variable_completeness(agents)


def create_empty_agent(name):
    return AbstractAgent(name=name)


@timeit
def main():
    f = '../inputs/agent.tex'
    A = LinkedAgent.read_from_tex(f)
    A.process()
    B = LinkedAgent.from_abstract(create_empty_agent('B'))
    C = LinkedAgent.from_abstract(create_empty_agent('C'))
    A.add_flow(Flow(A, C, 3, 'rub'))
    B.add_flow(Flow(A, B, 6, 'tv'))
    print(A.__class__())
    print(A.name)
    print(A.Lagrangian)
    print(A.lagrangian)
    print(A.euler_equations())
    print(A.transversality_conditions())
    print(A.control_optimality())
    print(A.KKT())
    print(A.print_flows())


if __name__ == "__main__":
    main()
