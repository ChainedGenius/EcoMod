from sympy import Symbol, Function, Expr

from datamodel.efunction import EFunction
from datamodel.parameter import Parameter


def refine_object(obj):
    # aka ecomodify
    if issubclass(type(obj), Symbol):
        return Parameter.from_symbol(obj)
    if issubclass(type(obj), Function):
        return EFunction.from_function(obj)
    if issubclass(type(obj), Expr):
        return obj
