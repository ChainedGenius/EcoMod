from numpy.random import rand
from sympy import Symbol, Integral, Derivative, Eq, Expr
from sympy import symbols, sympify, sin, gamma, evaluate, Mul
from sympy.core.function import AppliedUndef, Function, FunctionClass

from task_classifier.classifier import ScalarInfo
from utils import iterable_substract


class CompletenessValidator:
    def __init__(self, expressions, variables_info):
        used_variables = set()
        variables = [*variables_info.keys()]
        for rel in expressions:
            used_variables.update(rel.atoms(Symbol, AppliedUndef))
            # used_variables.update({f.func for f in rel.atoms(AppliedUndef)})

        if iterable_substract(used_variables, variables):
            raise AssertionError(f'There is more variables used than initialized: {iterable_substract(used_variables, variables)}')

        if iterable_substract(variables, used_variables):
            raise AssertionError(f'There is more inited variables than used in expressions: {iterable_substract(variables, used_variables)}')


class ExpressionDimensionValidator:
    ultimate_dimension = sympify('X')
    zero_dimension = sympify(1)

    def __init__(self, relations, variables_info, deep=False):

        random_coef = rand(len(variables_info, ))
        var2dim = {kv[0]: kv[1].dim * random_coef[i] for i, kv in enumerate(variables_info.items())}

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
                dim_rel = buf_rel.subs(var2dim)

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


if __name__ == '__main__':
    # input
    a, b, c, d, t = symbols('a b c d t')
    x, y, z = symbols('x y z', cls=Function)
    # test
    CompletenessValidator([sin(a) + b + gamma(c) + x(t) + y(t), y(t) + x(t)], [a, b, c, x(t), y(t), t])
    ExpressionDimensionValidator([Eq(d * (sin(sin(a)) + gamma(b)), x(t).diff(t)), Eq(Integral(y(t), (t, 0, 1)), c)],
                                 {
                                     a: ScalarInfo(dim=sympify(1), desc=''),
                                     b: ScalarInfo(dim=sympify(1), desc=''),
                                     c: ScalarInfo(dim=sympify('rub*year'), desc=''),
                                     d: ScalarInfo(dim=sympify('rub'), desc=''),
                                     x(t): ScalarInfo(dim=sympify('rub*year'), desc=''),
                                     y(t): ScalarInfo(dim=sympify('rub'), desc=''),
                                     t: ScalarInfo(dim=sympify('year'), desc=''),
                                 }, deep=True)


