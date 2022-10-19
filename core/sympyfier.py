import sympy
from sympy import Function, Symbol, Number
from sympy.core.relational import Relational, Eq, Unequality
from sympy.parsing.latex import parse_latex
from utils import extract_dim_desc, set_equality
from datamodel import Parameter, Phase, Boundary
from errors.RWErrors import NonSympyfiableError, VariableAmbiguity
from itertools import chain
import numpy as np


def _xsympify(raw_str):
    """
    :param raw_str: raw sympy string
    :return:
    """
    decoded = parse_latex(raw_str)
    return decoded


def sympify(raw_obj):
    decoded = _xsympify(raw_obj[0])
    dim, desc = extract_dim_desc(raw_obj[1])
    if decoded.args:
        # case if not Symbol
        if decoded.__str__() == raw_obj[0]:
            # case if Function
            decoded = Phase(decoded.name, *decoded.args, dim=dim, desc=desc)
        elif issubclass(decoded.__class__, Relational):
            # case if Relation
            decoded = decoded

        else:
            raise NonSympyfiableError(err=decoded.__str__)
    else:
        # case if Symbol
        decoded = Parameter(decoded.name, dim=dim, desc=desc)

    return decoded


def ecomodify(raw_model):
    def sorter(i, e, f, p, object):
        if object.__class__ == Parameter:
            p.append(object)
        elif issubclass(object.__class__, Function):
            f.append(object)
        elif issubclass(object.__class__, Eq):
            e.append(object)
        elif issubclass(object.__class__, Relational):
            i.append(object)
        else:
            pass
        return i, e, f, p

    def find_analog(var1, evars):
        """
        :param var1: true sympy variables
        :param evars: set of real ecomod variables
        :return: analog in evars| var1.name = evars.name
        """
        try:
            res = [i for i in evars if i.name == var1.name]
        except:
            if var1.args:
                args_converted = [find_analog(i, evars) for i in var1.args]
                if args_converted:
                    return var1.func(*args_converted)
                else:
                    return var1.func(*var1.args)
            return var1
        if len(res) == 1:
            res = res[0]
            if var1.args:
                args_converted = [find_analog(i, evars) for i in var1.args]
                if args_converted:
                    res = res.func(*args_converted)
                else:
                    res = res.func(*var1.args)
            return res
        else:
            raise VariableAmbiguity(var1=res[0], var2=res[1])

    inequations = []
    equations = []
    functions = []
    params = []
    for o in raw_model.items():
        inequations, equations, functions, params = sorter(inequations, equations, functions, params, sympify(o))
    # free_symbols soft checking
    # used = []
    # used_old = []
    # for eq in equations:
    #     used_old.extend(eq.atoms(sympy.Symbol))
    #     used_old.extend(eq.atoms(sympy.Function))
    #     for fs in eq.free_symbols.union(eq.find(Function)):
    #         buf = find_analog(fs, functions + params)
    #         eq = eq.replace(fs, buf)
    #     used.extend(eq.atoms(sympy.Symbol))
    #     used.extend(eq.atoms(sympy.Function))
    # print(list(set(used_old)))
    # print(list(set(used)))
    # print(functions + params)
    fs_all = set(chain(*[eq.free_symbols.union([f.simplify() for f in eq.atoms(Function)]) for eq in equations]))
    fs_all_new = [find_analog(fs, functions + params) for fs in fs_all]
    fs_map = {fs: find_analog(fs, functions + params) for fs in fs_all}
    # print(fs_all_new)
    # test1 = set(chain(*[v.args for v in fs_all_new]))
    # for i in test1:
    #     if i.args:
    #         print(i.args[0].__class__,i.__class__,i.args)

    # xreplacing
    functions = [f.subs(fs_map) for f in functions]
    params = [p.subs(fs_map) for p in params]
    equations = [e.subs(fs_map) for e in equations]
    inequations = [i.subs(fs_map) for i in inequations]

    # compatibility testing
    # func porting tests
    fs_all_ = set(chain(*[eq.free_symbols.union([f.simplify() for f in eq.atoms(Function)]) for eq in equations]))
    test1 = set([i.func for i in fs_all_ if i.func])
    test2 = set([i.func for i in params + functions if i.func])
    if not set_equality(test1, test2):
        raise TypeError("Ecomodify problems")
    # args porting tests
    test1 = set(chain(*[i.args for i in fs_all_ if i.args]))
    test2 = set([i for i in params + functions])
    numbersDOTtk = test1 - test2  # cicada meme
    if np.prod([issubclass(i.__class__, Number) for i in numbersDOTtk]) == 0:
        raise TypeError("Ecomodify problems")

    return inequations, equations, functions, params
