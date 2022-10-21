import random
from itertools import chain

from sympy import sympify, Expr, Function, sinh, cosh, tanh, exp, log, Derivative, symbols, simplify, Eq
from sympy.parsing.latex import parse_latex
from sympy import sin, cos, tan, cot, sinh, cosh, tanh, coth, exp, log
from multipledispatch import dispatch
from sympy import GreaterThan

def KKT_mask(dual: dict):
    return [*chain(*[(GreaterThan(v, 0), Eq(k*v, 0)) for k, v in dual.items()])]

def euler_mask(L, x, t):
    if x.args and t in x.args:
        x = x.func
    else:
        raise TypeError('Wrong time :)')
    x_prime = x(t).diff(t)
    L_x_prime = L.diff(x_prime)
    L_x = L.diff(x(t))
    return simplify(Derivative(L_x_prime, t) - L_x)


def transversality_mask(L, x, t, l, t0, t1):
    if x.args and t in x.args:
        x = x.func
    else:
        raise TypeError('Wrong time :)')
    # lhs
    x_prime = x(t).diff(t)
    L_x_prime = L.diff(x_prime)
    # rhs
    l_x_t_0 = l.diff(x(t).subs({t: t0}))
    l_x_t_1 = -l.diff(x(t).subs({t: t1}))
    return Eq(L_x_prime.simplify().subs({t: t0}), l_x_t_0), Eq(L_x_prime.simplify().subs({t: t1}), l_x_t_1)


def generate_symbols(tag, count, cls):
    query = [tag + "_" + str(i) + " " for i in range(count)]
    query = ''.join(query)[:-1]
    return symbols(query, cls=cls)


def spec_funcs():
    return {sin, cos, tan, cot, sinh, cosh, tanh, coth, exp, log}


def is_spec_function(func):
    spec = {sin, cos, tan, cot, sinh, cosh, tanh, coth, exp, log}
    return func.__class__ in spec


def eq2func(eq):
    return eq.args[0] - eq.args[1]


def deriv_degree(bc):
    deg = 0
    if eq2func(bc).class_key()[-1] == 'Derivative':  ## catch derivative of x eqs 0
        deg_ = sum(i[-1] for i in eq2func(bc).variable_count)
        return deg_
    for c in eq2func(bc).args:
        # print(c.class_key()[-1])
        deg_ = 0
        if c.class_key()[-1] == 'Derivative':
            deg_ = sum([i[-1] for i in c.variable_count])

        if c.class_key()[-1] == 'Mul':
            for m in c.args:
                if m.class_key()[-1] == 'Derivative':
                    deg_ = sum(i[-1] for i in m.variable_count)

        if deg_ > deg:
            deg = deg_

    return deg


@dispatch(dict)
def span(d: dict):
    from sympy.core.numbers import Zero
    ret = Zero()
    for k, v in d.items():
        ret += k * v

    return ret


@dispatch(set, set)
@dispatch(list, list)
def span(coefs, variables):
    if len(coefs) != len(variables):
        raise TypeError

    ret = 0
    for i, j in zip(coefs, variables):
        ret += i * j

    return ret


def pi_theorem(vars, eq):
    ret = True  # bool return

    def _dim_subs(vars, eq):
        for k, v in vars.items():
            eq_ = eq.replace(sympify(k), sympify(v))
            if 0 not in eq_.args:
                eq = eq_
            else:
                coef = random.random()
                eq = eq.replace(sympify(k), sympify(v + '*' + str(coef)))

        # print(eq)
        # eq = eq.subs(vars).simplify()
        if eq.class_key()[-1] == 'Equality':
            lhs = eq.args[0]
            rhs = eq.args[1]
            if (lhs * rhs ** (-1)).class_key()[-1] == 'Number':
                return True
            else:
                return False
        else:
            if eq.simplify().class_key()[-1] == 'Number':
                return True
            else:
                return False

    # var correctance
    vars_ = {k: Expr(v) for k, v in vars.items()}
    eq = parse_latex(eq)
    expr = eq2func(eq)
    # parse functions
    funcs = [*expr.atoms(Function)]
    # check correctance

    actual_vars = [str(v) for v in expr.free_symbols]
    actual_vars.extend([f.class_key()[-1] for f in funcs])
    # if set(actual_vars) != set(vars.keys()):
    #    raise Warning('check var_list isnt full', set(actual_vars)- set(vars.keys()))

    # parse special functions
    special_tags = [sinh, cosh, tanh, exp, log]
    occ_funcs = []
    occurencies = []
    for tag in special_tags:
        occ_funcs.extend([*expr.find(tag)])
        occurencies.extend([e.args[0] for e in expr.find(tag)])

    for expr_ in occurencies:
        ret = _dim_subs(vars, expr_)
        if ret is False:
            return False

    # swap special functions
    for of in occ_funcs:
        eq = eq.replace(of, 1)

    # parse derivatives
    derivatives = [*expr.find(Derivative)]

    # swap derivas
    for d in derivatives:
        var, order = d.args
        buf = var / order[0] ** order[1]
        eq = eq.replace(d, buf)

    # swap functions
    for f in funcs:
        eq = eq.replace(f, symbols(f.class_key()[-1]))

    ret = _dim_subs(vars, eq)
    if ret is False:
        return False
    return ret
