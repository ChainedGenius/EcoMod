from sympy import sympify
from sympy.core._print_helpers import Printable
from sympy.core.function import AppliedUndef
from types import FunctionType

class Config(dict):
    def __init__(self, **kwargs):
        # self.__validate_keys(kwargs)
        kwargs = self.__sympify_values(kwargs)
        super().__init__(kwargs.items())
        self.unfilled = self.__find_unfilled()


    # def __validate_keys(self, kwargs):
    #     has_bad_keys = any(not issubclass(k.__class__, Printable) for k in kwargs)
    #     if has_bad_keys:
    #         raise TypeError('Not all keys are Sympy objects, but have to!')
    @staticmethod
    def __sympify_values(kwargs):
        return {sympify(k): sympify(v) for k, v in kwargs.items()}

    def __find_unfilled(self):
        pass

    @classmethod
    def unfilled_from_list(cls, keys):
        return cls.__init__(**{k: k for k in keys})

    def __getitem__(self, key):
        return super().__getitem__(sympify(key))

    def __setitem__(self, key, value):
        if (issubclass(sympify(key).__class__, Symbol)
                and issubclass(sympify(value).__class__, Symbol)):
            super().__setitem__(sympify(key), sympify(value))
        if (issubclass(sympify(key).__class__, Symbol)
                and issubclass(sympify(value).__class__, Number)):
            super().__setitem__(sympify(key), sympify(value))
        elif (issubclass(sympify(key).__class__, AppliedUndef)
              and issubclass(value.__class__, FunctionType)):
            super().__setitem__(sympify(key), value)
        elif (issubclass(sympify(key).__class__, AppliedUndef)
              and issubclass(sympify(value).__class__, (AppliedUndef, FunctionClass))):
            super().__setitem__(sympify(key), sympify(value))

        else:
            raise TypeError(f'Incompatible key and value {sympify(key).__class__} and {sympify(value).__class__}')




if __name__ == '__main__':
    from sympy import *
    t = Symbol('t')
    x = Function('x')
    a = Config(k='k', m=2)
    print(a)

    j,b,c = symbols('a b c')
    func = lambdify([j,b,c], j+b+c)
    a['k(t)'] = func

    print(a)