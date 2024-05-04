from sympy import sympify, Number, Symbol
from sympy.core.function import AppliedUndef, FunctionClass
from types import FunctionType


class Config(dict):
    def __init__(self, **kwargs):
        kwargs = self.__sympify_values(kwargs)
        super().__init__(kwargs.items())

    @staticmethod
    def __sympify_values(kwargs):
        return {sympify(k): sympify(v) for k, v in kwargs.items()}

    @classmethod
    def unfilled_from_list(cls, keys):
        return cls.__init__(**{k: k for k in keys})

    def __getitem__(self, key):
        return super().__getitem__(sympify(key))

    def __setitem__(self, key, value):
        if value is None and key is not None:
            super().__setitem__(sympify(key), sympify(key))

        elif (issubclass(sympify(key).__class__, Symbol)
                and issubclass(sympify(value).__class__, Symbol)):
            super().__setitem__(sympify(key), sympify(value))

        elif (issubclass(sympify(key).__class__, Symbol)
              and issubclass(sympify(value).__class__, Number)):
            super().__setitem__(sympify(key), sympify(value))

        elif (issubclass(sympify(key).__class__, AppliedUndef)
              and issubclass(value.__class__, FunctionType)):
            super().__setitem__(sympify(key), value)

        elif (issubclass(sympify(key).__class__, AppliedUndef)
              and isinstance(sympify(value).__class__, FunctionClass)):
            super().__setitem__(sympify(key), sympify(value))

        else:
            raise TypeError(f'Incompatible key and value {sympify(key).__class__} and {sympify(value).__class__}')

    def is_filled(self):
        return False if self.unfilled_keys() else True

    def is_numeric(self):
        return True if all(issubclass(v.__class__, (Number, FunctionType)) for v in self.values()) else False

    def as_dict(self):
        return self.copy()

    def unfilled_keys(self):
        unfilled_keys = []
        for k, v in self.items():
            if k == v:
                unfilled_keys.append(k)
        return unfilled_keys
