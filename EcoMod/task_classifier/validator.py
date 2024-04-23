from types import FunctionType

from numpy.random import rand
from sympy import Symbol, Integral, Derivative, Expr
from sympy import sympify, evaluate, Mul
from sympy.core.function import AppliedUndef, Function

from EcoMod.utils import iterable_substract


class Validator:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class NumericConfigValidator(Validator):
    def __init__(self, config):
        super().__init__(config=config)
        errlist = []
        for k, v in config.items():
            if issubclass(sympify(k).__class__, Symbol):
                if not v.is_constant():
                    errlist.append((k, v))

            elif issubclass(sympify(k).__class__, AppliedUndef):
                if not issubclass(v.__class__, FunctionType):
                    errlist.append((k, v))
            else:
                errlist.append((k, v))

        if errlist:
            raise TypeError(f'Some of values are not computable expressions {errlist}')

class DifferentialAlgebraicValidator(Validator):
    def __init__(self, exprs):
        super().__init__(exprs=exprs)
        errlist = []
        for e in exprs:
            if e.find(Integral):
                errlist.append(e)

        if errlist:
            raise AssertionError(f'There are some integral operators or algebraic equations here {errlist}')


class DifferentialValidator(Validator):
    def __init__(self, exprs):
        super().__init__(exprs=exprs)
        errlist = []
        for e in exprs:
            if e.find(Integral) or not e.find(Derivative):
                errlist.append(e)

        if errlist:
            raise AssertionError(f'There are some integral operators or algebraic equations here {errlist}')


class AlgebraicValidator(Validator):
    def __init__(self, exprs):
        super().__init__(exprs=exprs)
        errlist = []
        for e in exprs:
            if e.find(Derivative) or e.find(Integral):
                errlist.append(e)

        if errlist:
            raise AssertionError(f'There are some differential or integral operators here {errlist}')


class CompletenessValidator(Validator):
    def __init__(self, expressions, variables_info):
        super().__init__(expressions=expressions, variables_info=variables_info)
        used_variables = set()
        variables = [*variables_info.keys()]
        for rel in expressions:
            used_variables.update(rel.atoms(Symbol, AppliedUndef))
            # used_variables.update({f.func for f in rel.atoms(AppliedUndef)})

        if iterable_substract(used_variables, variables):
            raise AssertionError(
                f'There is more variables used than initialized: {iterable_substract(used_variables, variables)}')

        if iterable_substract(variables, used_variables):
            raise AssertionError(
                f'There is more inited variables than used in expressions: {iterable_substract(variables, used_variables)}')


class ExpressionDimensionValidator(Validator):
    ultimate_dimension = sympify('X')
    zero_dimension = sympify(1)

    def __init__(self, relations, variables_info, deep=False):
        super().__init__(relations=relations, variables_info=variables_info)
        random_coef = rand(len(variables_info, ))
        var2dim = {kv[0] if isinstance(kv[0], Symbol) else Symbol(kv[0].func.name): kv[1].dim * random_coef[i] for i, kv
                   in enumerate(variables_info.items())}

        errors = []
        spec_stack = set()
        for rel in relations:
            if isinstance(rel, Expr):
                continue
            spec_funcs = {i for i in rel.atoms(Function) - rel.atoms(AppliedUndef)}
            spec_stack.update(i.args[0] for i in spec_funcs)
            with evaluate(False):
                buf_rel = rel.subs({i: self.zero_dimension * rand(1)[0] for i in spec_funcs})

                # deal with integrals
                buf_rel = buf_rel.subs({i: i.args[0] * i.args[1][0] for i in buf_rel.find(Integral)})

                # deal with derivatives
                buf_rel = buf_rel.subs(
                    {i: i.args[0] / Mul(*[der[0] * der[1] for der in i.args[1:]]) for i in buf_rel.find(Derivative)})

                # substitute randomized dictionary
                dim_rel = buf_rel.subs({k: Symbol(k.func.name) for k in buf_rel.find(AppliedUndef)})
                dim_rel = dim_rel.subs(var2dim)
                # print(dim_rel)
                # print('\n')
            if not (dim_rel.rhs / dim_rel.lhs).simplify().is_constant():
                errors.append(rel)
        if deep:
            spec_stack = list(spec_stack)
            i = 0
            while i < len(spec_stack):
                expr = spec_stack[i]
                spec_funcs = {i for i in expr.atoms(Function) - expr.atoms(AppliedUndef)}
                spec_stack.extend(i.args[0] for i in spec_funcs)
                with evaluate(False):
                    expr_rel = expr \
                        .subs({i: self.zero_dimension * rand(1)[0] for i in spec_funcs}) \
                        .subs({i: i.args[0] * i.args[1][0] for i in buf_rel.find(Integral)}) \
                        .subs({i: i.args[0] / Mul(*[der[0] * der[1] for der in i.args[1:]]) for i in
                               buf_rel.find(Derivative)}) \
                        .subs(var2dim)

                if not expr_rel.is_constant():
                    errors.append(expr)
                i += 1

        if errors:
            error_string = '\n'.join([i.__str__() for i in errors])
            raise ArithmeticError(f'Wrong dimension in those expressions: {error_string}')


class OptimalControlProblemValidator:
    def __new__(cls, validators=None, **kwargs):
        if validators is None:
            validators = [CompletenessValidator, ExpressionDimensionValidator]

        for v in validators:
            v(kwargs['equations'] + kwargs['inequations'] + kwargs['maximizations'], kwargs['variable_info'])

        name = kwargs.get('name', '')
        objectives = kwargs.get('maximizations', [])
        inequations = kwargs.get('inequations', [])
        equations = kwargs.get('equations', [])
        functions = kwargs.get('functions', [])
        params = kwargs.get('symbols', [])
        dim_dict = kwargs.get('variable_info', {})
        external_functions = kwargs.get('external_functions', [])
        config = kwargs.get('config', {})
        return {
            'name': name,
            'objectives': objectives,
            'inequations': inequations,
            'equations': equations,
            'functions': functions,
            'params': params,
            'dim_dict': dim_dict,
            'external_functions': external_functions,
            'config': config
        }
