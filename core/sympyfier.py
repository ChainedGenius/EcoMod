from sympy import Function, Symbol
from sympy.core.relational import Relational, Eq, Unequality
from sympy.parsing.latex import parse_latex
from utils import extract_dim_desc
from datamodel import Parameter, Phase, Boundary
from errors.RWErrors import NonSympyfiableError


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
        print(var1)
        try:
            res = [i for i in evars if i.name == var1.name]
        except:
            print(var1)
            return Symbol('XXX')
        if len(res) == 1:
            return res[0]
        else:
            raise TypeError('alo ept')

    inequations = []
    equations = []
    functions = []
    params = []
    for o in raw_model.items():
        inequations, equations, functions, params = sorter(inequations, equations, functions, params, sympify(o))
    # free_symbols soft checking
    used = []
    for eq in equations:
        for fs in eq.free_symbols.union(eq.find(Function)):
            buf = find_analog(fs, functions + params)
            eq = eq.replace(fs, buf)
        used.extend(eq.free_symbols)
    print(list(set(used)))
    print(functions + params)
    return inequations, equations, functions, params
