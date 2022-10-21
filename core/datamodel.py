from sympy import Function, Symbol


class Parameter(Symbol):  # superclass for ECOMOD params
    def __new__(cls, name, **kwargs):
        obj = Symbol.__xnew__(cls, name)
        obj.__dict__.update(kwargs)
        return obj


class Phase(Function):
    def __new__(cls, name, dim, desc, *args):
        obj = Function(name)(*args)
        # TODO: add dim desc as attributes of class
        return obj

    # def __new__(cls, name, *args, **kwargs):
    #     obj = Symbol.__xnew__(cls, name)
    #     obj.dim = kwargs["dim"]
    #     obj.desc = kwargs["desc"]
    #     obj = Function(obj)(*args)  # TODO: CHECK WHY functions in sympyfier == []
    #     return obj

    # def __init__(self, name, *args, **kwargs):
    #     self.obj = Function(name)(*args)
    #     self.__dict__.update(kwargs)

    # @property
    # def dim(self):
    #     self.__dict__.update({'dim': self.dim})
    #     return self.dim
    #
    # @dim.setter
    # def dim(self, dim):
    #     if isinstance(dim, str):
    #         self.dim = dim
    #         self.__dict__.update({'dim': self.dim})
    #     else:
    #         raise TypeError('Dimension must be a string')
    #
    # @property
    # def desc(self):
    #     self.__dict__.update({'desc': self.desc})
    #     return self.desc
    #
    # @desc.setter
    # def desc(self, desc):
    #     if isinstance(desc, str):
    #         self.desc = desc
    #         self.__dict__.update({'desc': self.desc})
    #     else:
    #         raise TypeError('Dimension must be a string')


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
