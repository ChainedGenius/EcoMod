import sympy
from sympy import Function, Symbol, Number, FunctionClass, sympify as real_sympify
from sympy.core.relational import Relational, Eq, Unequality
from sympy.parsing.latex import parse_latex

from ecomod_utils import spec_funcs, is_spec_function
from utils import extract_dim_desc, set_equality, find_objective_markers, iterable_substract
from datamodel import Parameter, Phase, Boundary
from errors.RWErrors import NonSympyfiableError, VariableAmbiguity, ExtraVariableError, DimensionInExpression
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
    raw_latex = raw_obj[0]
    is_objective = find_objective_markers(raw_obj[0])
    if is_objective[0]:
        raw_latex = is_objective[1]
    decoded = _xsympify(raw_latex)
    dim, desc = extract_dim_desc(raw_obj[1])
    if dim == "":
        dim = 1
    if decoded.args:
        # case if not Symbol
        if decoded.__str__() == raw_latex:
            # case if Function
            # decoded = Phase(decoded.name, *decoded.args, dim=dim, desc=desc)
            decoded = Phase(decoded.name, dim, desc, *decoded.args)
            # decoded.dim = dim
            # decoded.desc = desc
        elif issubclass(decoded.__class__, Relational):
            # case if Relation
            decoded = decoded

        else:
            raise NonSympyfiableError(err=decoded.__str__)
    else:
        # case if Symbol
        decoded = Parameter(decoded.name, dim=dim, desc=desc)

    return decoded, is_objective[0]


def ecomodify(raw_model):
    def dim_dict(_raw_model):
        # not real python dict: list of tuples
        return [(k, real_sympify(extract_dim_desc(v)[0])) if extract_dim_desc(v)[0] != "" else (k, "") for k, v in _raw_model.items()]

    def sorter(i, e, f, p, o, objecti):
        object, is_objective = objecti
        if is_objective:
            o.append(object)
        elif object.__class__ == Parameter:
            p.append(object)

        elif issubclass(object.__class__, Function):
            f.append(object)

        elif issubclass(object.__class__, Eq):
            e.append(object)

        elif issubclass(object.__class__, Relational):
            i.append(object)

        else:
            pass

        return i, e, f, p, o

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
    objectives = []
    dim_dict = dim_dict(raw_model)
    for o in raw_model.items():
        inequations, equations, functions, params, objectives = sorter(inequations, equations, functions, params,
                                                                       objectives, sympify(o))
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
    fs_all = set(chain(*[eq.free_symbols.union([f.simplify() for f in eq.atoms(Function)]) for eq in
                         equations + inequations + objectives]))
    fs_all_new = [find_analog(fs, functions + params) for fs in fs_all]
    fs_map = {fs: find_analog(fs, functions + params) for fs in fs_all}
    # print(fs_all_new)
    # test1 = set(chain(*[v.args for v in fs_all_new]))
    # for i in test1:
    #     if i.args:
    #         print(i.args[0].__class__,i.__class__,i.args)

    # xreplacing

    functions = [f.xreplace(fs_map) for f in functions]
    params = [p.xreplace(fs_map) for p in params]
    equations = [e.xreplace(fs_map) for e in equations]
    inequations = [i.xreplace(fs_map) for i in inequations]
    objectives = [o.xreplace(fs_map) for o in objectives]
    # v0 hack
    try:
        dim_dict = {fs_map[parse_latex(i[0])]: i[1] for i in dim_dict if i[1] != ''}
    except KeyError as exc:
        raise DimensionInExpression(expr=exc)

    # functions = [f.subs(fs_map) for f in functions]
    # params = [p.subs(fs_map) for p in params]
    # equations = [e.subs(fs_map) for e in equations]
    # inequations = [i.subs(fs_map) for i in inequations]
    # objectives = [o.subs(fs_map) for o in objectives]

    # compatibility testing
    # func porting tests
    # ----------- UNCOMMENT -------------------------
    fs_all_ = set(chain(*[eq.free_symbols.union([f.simplify() for f in eq.atoms(Function)]) for eq in
                          equations + inequations + objectives]))
    test1 = iterable_substract(set([i.func for i in fs_all_ if i.func]), spec_funcs())
    test2 = set([i.func for i in params + functions if i.func])
    if not set_equality(test1, test2):
        print(test1, test2)
        raise TypeError("Ecomodify problems")
    # args porting tests
    test1 = set(chain(*[i.args if np.prod([k.is_Function or k.is_symbol for k in i.args]) else i.atoms() for i in fs_all_ ]))
    test2 = set([i for i in params + functions])
    numbersDOTtk = test1 - test2  # cicada meme
    if np.prod([issubclass(i.__class__, Number) for i in numbersDOTtk]) == 0:
        print(test1, test2)
        raise TypeError("Ecomodify problems")

    # completeness
    completion_names = set(j.name for j in chain(*[i.atoms(Function).union(i.atoms(Symbol)) for i in fs_all_]) if not is_spec_function(j))
    inited_names = set(j.name for j in functions + params)
    if completion_names != inited_names:
        raise ExtraVariableError(vars=completion_names - inited_names)
    # ---------------------UNCOMMENT-----------------
    return objectives, inequations, equations, functions, params, dim_dict
