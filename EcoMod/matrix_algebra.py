from itertools import chain

from sympy import *
from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter


def matrices(names):
    ''' Call with  A,B,C = matrix('A B C') '''
    return symbols(names, commutative=False)


# Transformations

d = Function("d", commutative=False)
inv = Function("inv", commutative=False)


class t(Function):
    """ The transposition, with special rules
        t(A+B) = t(A) + t(B) and t(AB) = t(B)t(A) """
    is_commutative = False

    def __new__(cls, arg):
        if arg.is_Add:
            return Add(*[t(A) for A in arg.args])
        elif arg.is_Mul:
            L = len(arg.args)
            return Mul(*[t(arg.args[L - i - 1]) for i in range(L)])
        else:
            return Function.__new__(cls, arg)


# Differentiation

def shape_dispatcher(e):
    if not e.is_Matrix or (e.rows == 1 and e.cols == 1):
        return 'scalar', e if e.is_Matrix else Matrix(1, 1, [e])
    elif e.rows == 1 or e.cols == 1:
        return 'vector', e
    elif e.rows > 1 and e.cols > 1:
        return 'matrix', e
    else:
        raise TypeError('Unknown object, not scalar, vector or matrix')


def common_matrix_diff(e, s):
    e_type, e = shape_dispatcher(e)
    s_type, s = shape_dispatcher(s)

    print(e_type, s_type)
    # matrix over matrix, for equal size matrices, s should be argument matrix
    if e_type == 'matrix' and s_type == 'matrix':
        return Matrix.__new__(type(e), *e.shape, [f.diff(x) for f, x in zip(e.values(), s.values())]) \
            if e.shape == s.shape else TypeError(f'Matrices should be equal size. ARG1: {e.shape}, ARG2: {s.shape}')

    if e_type == 'matrix' and s_type == 'vector':
        return Matrix.__new__(type(e), e.shape[0], s.shape[1],
                              [Add(*[f.diff(x) for f, x in zip(e[i, :].values(), s.values())]) for i in range(e.rows)]) \
            if e.shape[1] == s.shape[0] else TypeError(
            f'Matrices should be multiplicative. ARG1: {e.shape}, ARG2: {s.shape}')

    if e_type == 'matrix' and s_type == 'scalar':
        return e.diff(s[0, 0])

    if e_type == 'vector' and s_type == 'matrix':
        raise NotImplementedError('Vector cannot be differentiated wrt matrix')

    if e_type == 'vector' and s_type == 'vector':
        return Matrix(len(e.values()), len(s.values()), [*chain(*[e.diff(x) for x in s.values()])])

    if e_type == 'vector' and s_type == 'scalar':
        return e.diff(s[0, 0])

    if e_type == 'scalar' and s_type == 'matrix':
        return e[0, 0].diff(s)

    if e_type == 'scalar' and s_type == 'vector':
        return e[0, 0].diff(s)

    if e_type == 'scalar' and s_type == 'scalar':
        return Matrix(1, 1, [e[0, 0].diff(s[0, 0])])

    # assert e.shape[1] == s.shape[0]
    # res = []
    # vars = s.values()
    # for i in range(e.rows):
    #     row = e[i, :].values()
    #     res.append(Add(*[f.diff(x) for f, x in zip(row, vars)]))
    #
    # return Matrix(e.shape[0], s.shape[1], res)


MATRIX_DIFF_RULES = {

    # e =expression, s = a list of symbols respect to which
    # we want to differentiate

    ImmutableDenseMatrix: lambda e, s: common_matrix_diff(e, s),
    MatAdd: lambda e, s: MatAdd(*[matDiff(arg, s) for arg in e.args]),
    MatMul: lambda e, s: MatMul(matDiff(e.args[0], s), MatMul(*e.args[1:]).T)
                         + MatMul(e.args[0], matDiff(MatMul(*e.args[1:]), s)),
    # t: lambda e, s: t(matDiff(e.args[0], s)),
    # inv: lambda e, s: - e * matDiff(e.args[0], s) * e
}


def matDiff(expr, symbols):
    if expr.__class__ in MATRIX_DIFF_RULES.keys():
        return MATRIX_DIFF_RULES[expr.__class__](expr, symbols)
    else:
        return common_matrix_diff(expr, symbols)


if __name__ == '__main__':
    t, a, b, c, d = symbols('t a b c d', cls=Symbol)
    f, u, x = symbols('f u x', cls=Function)
    f = f(t)
    u = u(t)
    x = x(t)
    A = Matrix(2, 3, [a, b, x, u * log(u), f, c])
    B = Matrix(2, 3, [t, x, t, u, f, c])
    V = Matrix(3, 1, [t, x, u])
    U = Matrix(3, 1, [t ** 2, x, u])
    print(U)
    print(V)
    res = matDiff(t, V)
    print(res)


# MATRIX_DIFF_RULES = {
#
#     # e =expression, s = a list of symbols respsect to which
#     # we want to differentiate
#
#     Symbol: lambda e, s: d(e) if (e in s) else 0,
#     Add: lambda e, s: Add(*[matDiff(arg, s) for arg in e.args]),
#     Mul: lambda e, s: Mul(matDiff(e.args[0], s), Mul(*e.args[1:]))
#                       + Mul(e.args[0], matDiff(Mul(*e.args[1:]), s)),
#     t: lambda e, s: t(matDiff(e.args[0], s)),
#     inv: lambda e, s: - e * matDiff(e.args[0], s) * e
# }
#
#
# def matDiff(expr, symbols):
#     if expr.__class__ in MATRIX_DIFF_RULES.keys():
#         return MATRIX_DIFF_RULES[expr.__class__](expr, symbols)
#     else:
#         return 0


#####  C  O  S  M  E  T  I  C  S


# Console mode

class matStrPrinter(StrPrinter):
    ''' Nice printing for console mode : X¯¹, X', ∂X '''

    def _print_inv(self, expr):
        if expr.args[0].is_Symbol:
            return self._print(expr.args[0]) + '¯¹'
        else:
            return '(' + self._print(expr.args[0]) + ')¯¹'

    def _print_t(self, expr):
        return self._print(expr.args[0]) + "'"

    def _print_d(self, expr):
        if expr.args[0].is_Symbol:
            return '∂' + self._print(expr.args[0])
        else:
            return '∂(' + self._print(expr.args[0]) + ')'


def matPrint(m):
    mem = Basic.__str__
    Basic.__str__ = lambda self: matStrPrinter().doprint(self)
    print(str(m).replace('*', ''))
    Basic.__str__ = mem


# Latex mode

class matLatPrinter(LatexPrinter):
    r''' Printing instructions for latex : X^{-1},  X^T, \partial X '''

    def _print_inv(self, expr):
        if expr.args[0].is_Symbol:
            return self._print(expr.args[0]) + '^{-1}'
        else:
            return '(' + self._print(expr.args[0]) + ')^{-1}'

    def _print_t(self, expr):
        return self._print(expr.args[0]) + '^T'

    def _print_d(self, expr):
        if expr.args[0].is_Symbol:
            return r'\partial ' + self._print(expr.args[0])
        else:
            return r'\partial (' + self._print(expr.args[0]) + ')'


def matLatex(expr, profile=None, **kargs):
    if profile is not None:
        profile.update(kargs)
    else:
        profile = kargs
    return matLatPrinter(profile).doprint(expr)
