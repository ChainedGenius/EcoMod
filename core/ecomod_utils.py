import random
from itertools import chain

from sympy import sympify, Expr, Function, sinh, cosh, tanh, exp, log, Derivative, symbols, simplify, Eq, Symbol
from sympy.parsing.latex import parse_latex
from sympy import sin, cos, tan, cot, sinh, cosh, tanh, coth, exp, log
from multipledispatch import dispatch
from sympy import GreaterThan, Basic
from sympy.printing.latex import latex


def is_substricted(symb, tag=None):
    """
    Check if symbol is substricted with tag.
    :param symb: Union[datamodel.Parameter, datamodel.Phase]
    :param tag: Union[string, None]
    :return: bool
    """
    # simple heuristics
    if not tag:
        return True if "_{" in symb.__str__() else False
    else:
        ret = True if f"_{{{tag}}}" in symb.__str__() else False
        return ret


def remove_subscript(symb):
    """
    Firstly check if symbol is substricted, then if true remove !ANY! substricted.
    :param symb: Union[datamodel.Parameter, datamodel.Phase]
    :return: Union[datamodel.Parameter, datamodel.Phase] -- unsubscripted.
    """
    if is_substricted(symb):
        if symb.args:
            # case if function
            return symbols(symb.name.split('_')[0], cls=Function)(*symb.args)
        return symbols(symb.name.split('_')[0], cls=Symbol)
    return symb


def add_subscript(symb, tag):
    """
    Add subscript `tag` to symbol
    :param symb: Union[datamodel.Parameter, datamodel.Phase] - untagged
    :param tag: str
    :return: Union[datamodel.Parameter, datamodel.Phase] -- tagged
    """
    if not is_substricted(symb):
        if symb.args:
            # case if function
            return symbols(f'{symb.name}_{{{tag}}}', cls=Function)(*symb.args)
        return symbols(f'{symb.name}_{{{tag}}}', cls=Symbol)
    return symb


def latexify(exprs: list, to_str=False):
    """
    Provide latex codesnippets to expressions. Can be returned as joined string or as List[str]
    :param exprs: List[Expr]
    :param to_str: bool
    :return: Union[List[Expr], str]
    """
    ret = [latex(e) for e in exprs]
    if not to_str:
        return ret

    return ',~'.join(ret)


def KKT_mask(dual: dict):
    """
    Returns Dual Feasibility and Complementary Slackness for dual dict
    :param dual: Dict[Union[datamodel.Parameter, datamodel.Phase] --> Expr]
    :return: List[Expr] = [K*V]
    """
    return [*chain(*[(GreaterThan(v, 0), Eq(k * v, 0)) for k, v in dual.items()])]


def euler_mask(L, x, t):
    """
    Provide Euler-Lagrange equations for passed Lagrangian and variables [phase, time]
    :param L: Expr -- Lagrangian
    :param x: datamodel.Phase -- Phase variables
    :param t: datamodel.Parameter -- Time variable
    :return: List[Expr]
    """
    if x.args and t in x.args:
        x = x.func
    else:
        raise TypeError('Wrong time :)')
    x_prime = x(t).diff(t)
    L_x_prime = L.diff(x_prime)
    L_x = L.diff(x(t))
    return Eq(simplify(Derivative(L_x_prime, t) - L_x), 0)


def transversality_mask(L, x, t, l, t0, t1):
    """
    Transveraslity condition for agent optimal control problem
    :param L: Expr -- Lagrangian
    :param x: datamodel.Phase -- phase variable
    :param t: datamodel.Parameter -- time variable
    :param l: Expr -- termination lagrangian
    :param t0: datamodel.Parameter -- time horizon[0]
    :param t1: datamodel.Parameter -- time horizon[1]
    :return: List[Eq]
    """
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
    """
    Generate symbols function. Used for dual variables and parameters generation in Agent.process
    :param tag: str -- Alias for generated symbols
    :param count: int -- count of symbols
    :param cls: Class[Union[datamodel.Parameter, datamodel.Phase]]
    :return: Iter[Symbol]
    """
    query = [tag + "_" + str(i) + " " for i in range(count)]
    query = ''.join(query)[:-1]
    return symbols(query, cls=cls) if count != 1 else [symbols(query, cls=cls)]


def spec_funcs():
    """
    Set of special analytical functions in sympy. Used to dimension check of its args.
    :return: set
    """
    return {sin, cos, tan, cot, sinh, cosh, tanh, coth, exp, log}


def is_spec_function(func):
    """
    Bool version of spec_funcs. Return true if func is listed in spec_funcs, otherwise else.
    :param func: function to be checked
    :return: bool
    """
    spec = {sin, cos, tan, cot, sinh, cosh, tanh, coth, exp, log}
    return func.__class__ in spec


def eq2func(eq):
    """
    Return Expr:= lhs - rhs from Eq
    :param eq: Eq to be transformed
    :return: Expr
    """
    return eq.args[0] - eq.args[1]


def deriv_degree(bc):
    """
    Returns maximum degree of differential operator inside `bc` expression.
    :param bc: Expr
    :return: int
    """
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


#@dispatch(dict)
def span_dict(d: dict):
    """
    Return linear span of KV-storage. ret = sum_0^N[K[i] * V[i]]
    :param d: Dict[Expr -> Expr]
    :return: Expr
    """
    from sympy.core.numbers import Zero
    ret = Zero()
    for k, v in d.items():
        ret += k * v

    return ret


#@dispatch(set, set)
#@dispatch(list, list)
def span(coefs, variables):
    """
    Return linear span of KV-storage. ret = sum_0^N[L_1[i] * L_2[i]]
    :param coefs: List[Expr]
    :param variables: List[Expr]
    :return: Expr
    """
    if len(coefs) != len(variables):
        raise TypeError

    ret = 0
    for i, j in zip(coefs, variables):
        ret += i * j

    return ret


def pi_theorem(vars, eq):
    """
    Deprecated method for dimension checking in equations.
    :param vars:
    :param eq:
    :return:
    """
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
