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
            decoded = Boundary(*decoded.args, decoded.rel_op, dim=dim, desc=desc)
        else:
            raise NonSympyfiableError(err=decoded.__str__)
    else:
        # case if Symbol
        decoded = Parameter(decoded.name, dim=dim, desc=desc)

    return decoded


def ecomodify(raw_model):
    def sorter(e, f, p, object):
        if object.__class__ == Parameter:
            p.append(object)
        if object.__class__ == Phase:
            f.append(object)
        if object.__class__ == Boundary:
            e.append(object)

        return e, f, p
    ret = []
    equations = []
    functions = []
    params = []
    for o in raw_model.items():
        equations, functions, params = sorter(equations, functions, params, sympify(o))
    # free_symbols soft checking
    return ret
