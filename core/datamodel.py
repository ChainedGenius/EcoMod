from sympy import Eq, Function, Symbol
from sympy.core.relational import Relational

class Parameter(Symbol):  # superclass for ECOMOD params
    def __new__(cls, name, **kwargs):
        obj = Symbol.__xnew__(cls, name)
        obj.__dict__.update(kwargs)
        return obj


class Phase(Function):
    def __new__(cls, name, *args, **kwargs):
        obj = Function(name)(*args)
        obj.__dict__.update(kwargs)
        return obj

class Boundary:
    pass
# class Boundary(Relational):
#     rel_op = "=="
#     def __new__(cls, lhs, rhs, rel_op, **kwargs):
#         obj = Relational.__new__(cls, lhs, rhs, rel_op)
#         #obj.__dict__ = {}
#         #obj.__dict__.update(kwargs)
#         return obj
#
# class IBoundary(Relational):
#     rel_op = "=="
#     def __new__(cls, lhs, rhs, rel_op, **kwargs):
#         obj = Relational.__new__(cls, lhs, rhs, rel_op)
#         #obj.__dict__ = {}
#         #obj.__dict__.update(kwargs)
#         return obj


if __name__ == "__main__":
    t = Parameter('t', dim=1, desc=2)
    c = Parameter('c', dim=2, desc=3)
    a = Phase('b', t, c, dim=1, desc=2)
    print(type(a.name))
    print(a.dim)
