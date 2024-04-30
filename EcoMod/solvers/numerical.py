import numpy as np
from sympy import Derivative, Symbol, Eq, Integer, nsolve, Add, symbols, Wild
from sympy.core.function import AppliedUndef

from EcoMod.datamodel.net import Net
from EcoMod.ecomod_utils import crawl, quantize
from EcoMod.utils import naive_extrapolation

FINITE_DIFF_DICT = {
    # i == current state
    4: lambda y, h, i: (y.func(i + 4) - 4 * y.func(i + 3) + 6 * y.func(i + 2) - 4 * y.func(i + 1) + y.func(i)) / (
            h ** 4),
    3: lambda y, h, i: (y.func(i + 3) - 3 * y.func(i + 2) + 3 * y.func(i + 1) - y.func(i)) / (h ** 3),
    2: lambda y, h, i: (y.func(i + 2) - 2 * y.func(i + 1) + y.func(i)) / (h ** 2),
    1: lambda y, h, i: (y.func(i + 1) - y.func(i)) / h,
    0: lambda y, h, i: y.func(i)
}


def general_numerical_solver(system, boundaries, tmin, tmax, N=50):
    def add_leading_nulls(s, N):
        # sorting help function
        return '0' * (N - len(s)) + s

    def variable_return(expr):
        return expr if expr in functions else expr.args[0]

    # Construct Net object
    net = Net(tmin, tmax, N)
    h = net.h
    t_array = net.as_array()

    # Collectors for give functions and orders of differential operators
    diff_variables = set()
    functions = set()
    diff_order_total = {}

    for eq in system:
        deriv_degree_dict = {i: i.args[1][1] for i in [*eq.find(Derivative)]}
        diff_variable = [i.args[0] for i in deriv_degree_dict.keys()]
        diff_variables.update(diff_variable)
        diff_order_total = diff_order_total | deriv_degree_dict
        functions.update(eq.atoms(AppliedUndef))

    diff_order_total = diff_order_total | {v: 0 for v in functions}

    # universal time variable definition
    tau = {i.args[0] for i in diff_variables}.pop()
    i = Symbol('i')  # TODO: Make more safe. Now it is unsafe if "i" figures in input derivatives or as "time" variable

    # Construct substitution dictionary which convert differential equations to finite difference equations
    linearized_subs = {k: FINITE_DIFF_DICT[v](variable_return(k), h, i) for k, v in diff_order_total.items()}
    linearized_subs |= {tau: tmin + (i - 1) * h}

    # Finite difference equations for each given differential equation
    linearized_system = [eq.subs(linearized_subs) for eq in system]

    # Construction of algebraic system of difference equations on given net
    full_system = []
    net_values_variables = set()

    left = 1
    right = N + max(diff_order_total.values())
    for k in range(left, right):
        for eq in linearized_system:
            linearized_i = eq.subs({i: k})
            full_system.append(linearized_i)
            net_values_variables.update(linearized_i.find(AppliedUndef))

    quant_subs = lambda expr: crawl(expr, quantize, net=net, i=i)
    for b in boundaries:
        deriv_degree_dict_boundary = {
            k: FINITE_DIFF_DICT[k.args[1][1]](variable_return(k), h, i)
            for k in [*b.find(Derivative)]
        }
        b_quant = b.subs(deriv_degree_dict_boundary)
        full_system.append(quant_subs(b_quant))

    # Additional sorting of net_values_variables for correct mapping below

    sorted_net_values_variables = sorted(net_values_variables,
                                         key=lambda i: i.name + '_' + add_leading_nulls(i.args[0].__str__(),
                                                                                        len(str(N))),
                                         reverse=False)
    vars_names = sorted(set([i.name for i in sorted_net_values_variables]), reverse=False)
    sols = nsolve(full_system, sorted_net_values_variables, [0 for _ in sorted_net_values_variables])
    mapping = {k: v for k, v in zip(sorted_net_values_variables, sols)}
    ret = {vars_names[i]: [v for k, v in mapping.items() if vars_names[i] in k.name][:N] for i
           in range(len(vars_names))}

    return t_array, ret
