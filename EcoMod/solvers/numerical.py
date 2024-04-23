import numpy as np
from sympy import Derivative, Function, Symbol, Eq, Integer, nsolve, Add, symbols, Wild
from sympy.core.function import AppliedUndef

FINITE_DIFF_DICT = {
    # i == current state
    4: lambda y, h, i: (y.func(i + 4) - 4 * y.func(i + 3) + 6 * y.func(i + 2) - 4 * y.func(i + 1) + y.func(i)) / (
            h ** 4),
    3: lambda y, h, i: (y.func(i + 3) - 3 * y.func(i + 2) + 3 * y.func(i + 1) - y.func(i)) / (h ** 3),
    2: lambda y, h, i: (y.func(i + 2) - 2 * y.func(i + 1) + y.func(i)) / (h ** 2),
    1: lambda y, h, i: (y.func(i + 1) - y.func(i)) / h,
    0: lambda y, h, i: y.func(i)
}

t = symbols('t', real=True)


def general_numerical_solver(system, boundaries, tmin, tmax, N=50):
    def add_leading_nulls(s, N):
        return '0' * (N - len(s)) + s

    def naive_extrapolation(l, N):
        return l + [l[-1] for _ in range(N - len(l))]

    def diff_degree_order(L, variables):
        pairs_var_order = [{i.args[0]: i.args[1][1]} for i in list(L.find(Derivative))]
        variables_and_order = []
        for k in variables:
            diff_order = []
            for j in range(len(pairs_var_order)):
                if list(pairs_var_order[j].keys())[0] == k:
                    diff_order.append([list(pairs_var_order[j].values())[0]])
            try:
                variables_and_order.append({k: max(diff_order)[0]})
            except ValueError:
                variables_and_order.append({k: 0})
        return variables_and_order

    h = (tmax - tmin) / N
    diff_variables = set()
    functions = set()
    diff_order_total = {}

    for eq in system:
        deriv_degree_dict = {i: i.args[1][1] for i in [*eq.find(Derivative)]}
        diff_order = max([*deriv_degree_dict.values()]) if len(deriv_degree_dict) > 0 else 0
        diff_variable = [i.args[0] for i in deriv_degree_dict.keys()]
        diff_variables.update(diff_variable)
        diff_order_total = diff_order_total | deriv_degree_dict
        functions.update(eq.atoms(AppliedUndef))

    vremya = {i.args[0] for i in diff_variables}.pop()
    tau = {i.args[0] for i in diff_variables}.pop()

    # diff order total aldready done for diffeqs, now add variables
    diff_order_total = diff_order_total | {v: 0 for v in functions}
    variables = list(functions)

    def rightmost_point_define(boundaries):
        high_index = []
        for i in range(len(boundaries)):
            if diff_degree_order(boundaries[i].lhs, [variables[0]])[0][variables[0]] >= 1:
                if len(boundaries[i].lhs.args) == 3:
                    deriv_order = diff_degree_order(boundaries[i].lhs, [variables[0]])[0][variables[0]]
                    high_index.append(Integer((boundaries[i].lhs.args[2][0] - tmin) / h + 1 + deriv_order))
                else:
                    der_order_in_bnd = []
                    for j in range(len(boundaries[i].lhs.args)):
                        der_order_in_bnd.append(
                            diff_degree_order(boundaries[i].lhs.args[j], [variables[0]])[0][variables[0]])
                        deriv_order = diff_degree_order(boundaries[i].lhs, [variables[0]])[0][variables[0]]
                    if len(boundaries[i].lhs.args[der_order_in_bnd.index(max(der_order_in_bnd))].args) == 3:
                        high_index.append(Integer((boundaries[i].lhs.args[
                                                       der_order_in_bnd.index(max(der_order_in_bnd))].args[2][
                                                       0] - tmin) / h + 1 + deriv_order))
                    if len(boundaries[i].lhs.args[der_order_in_bnd.index(max(der_order_in_bnd))].args) == 2:
                        high_index.append(Integer((boundaries[i].lhs.args[
                                                       der_order_in_bnd.index(max(der_order_in_bnd))].args[1].args[2][
                                                       0] - tmin) / h + 1 + deriv_order))
        if high_index == []:
            return 0
        else:
            max_high_index = max(high_index)
            return max_high_index

    # new arch
    i = Symbol('i')

    def gryaz(expr):
        return expr if expr in functions else expr.args[0]

    linearized_subs = {k: FINITE_DIFF_DICT[v](gryaz(k), h, i) for k, v in diff_order_total.items()}
    linearized_subs |= {vremya: tmin + i * h}
    linearized_system = [eq.subs(linearized_subs) for eq in system]
    full_system = []
    setka_variables = set()

    left = 1
    right = max([rightmost_point_define(boundaries) - diff_order + 1, N + 1 - diff_order + 1])
    for k in range(left, right):
        for eq in linearized_system:
            linearized_i = eq.subs(
                {i: k})  # .subs({b.lhs.func(Integer((b.lhs.args[0] - tmin)/h)+1): b.rhs for b in boundaries})
            full_system.append(linearized_i)
            setka_variables.update(linearized_i.find(AppliedUndef))

    def create_bound_match(dif, k, c, ddo):
        wild_n_order = dif * k
        ax_eq = Eq(bnd.lhs.match(wild_third_boundaries)[c], 0)
        q = Eq(ax_eq.lhs.match(wild_n_order)[k], 0)
        if ddo == 0:
            eq_to_add = ax_eq.lhs.match(wild_n_order)[dif] * (q.lhs.func(Integer((q.lhs.args[0] - tmin) / h) + 1))
        if ddo > 0:
            x = q.lhs.args[0].args[0]
            i = Integer((q.lhs.args[2][0] - tmin) / h + 1)
            if ddo == 1:
                eq_to_add = ax_eq.lhs.match(wild_n_order)[dif] * (x.subs(t, i + 1) - x.subs(t, i)) / h
            elif ddo == 2:
                eq_to_add = ax_eq.lhs.match(wild_n_order)[dif] * (
                            x.subs(t, i + 2) - 2 * x.subs(t, i + 1) + x.subs(t, i)) / h ** 2
            elif ddo == 3:
                eq_to_add = ax_eq.lhs.match(wild_n_order)[dif] * (
                            x.subs(t, i + 3) - 3 * x.subs(t, i + 2) + 3 * x.subs(t, i + 1) - x.subs(t, i)) / h ** 3
        return eq_to_add

    order_of_equation = diff_degree_order(system[0].lhs, [variables[0]])[0][variables[0]]
    for bnd in boundaries:
        c1, c2, c3, c4 = symbols('c1 c2 c3 c4', cls=Wild)
        a, b, c, d = symbols('a b c d', cls=Wild)
        wild_const = [c1, c2, c3, c4]
        wild_third_boundaries = c1 + c2 + c3 + c4
        output_equation = 0
        for c in wild_const:
            if str(Eq(bnd.lhs.match(wild_third_boundaries)[c], 0)) == 'True':
                pass
            else:
                ddo = diff_degree_order(Eq(bnd.lhs.match(wild_third_boundaries)[c], 0), [variables[0]])[0][variables[0]]
                for j in range(order_of_equation):
                    if ddo == j:
                        k, dif = symbols('k dif', cls=Wild)
                        output_equation = Add(output_equation, create_bound_match(dif, k, c, ddo))
        full_system.append(Eq(output_equation, bnd.rhs))

    sorted_setka_variables = sorted(setka_variables,
                                    key=lambda i: i.name + '_' + add_leading_nulls(i.args[0].__str__(), len(str(N))),
                                    reverse=False)
    vars_names = sorted(set([i.name for i in sorted_setka_variables]), reverse=False)
    sols = nsolve(full_system, sorted_setka_variables, [0 for _ in setka_variables])
    mapping = {k: v for k, v in zip(sorted_setka_variables, sols)}
    ret = {vars_names[i]: naive_extrapolation([v for k, v in mapping.items() if vars_names[i] in k.name], N + 1) for i
           in range(len(vars_names))}
    T = np.linspace(tmin, tmax, N + 1)

    return T, ret, vars_names
