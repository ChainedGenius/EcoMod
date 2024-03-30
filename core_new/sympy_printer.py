from sympy.core.symbol import Symbol
from sympy.printing.latex import LatexPrinter, print_latex
from sympy.printing.str import StrPrinter
from sympy.core.function import UndefinedFunction, Function


class LaTeXPrinter(LatexPrinter):
    """Print derivative of a function of symbols in a shorter form.
    """
    def _print_Derivative(self, expr):
        function, *vars = expr.args
        if not isinstance(type(function), UndefinedFunction) or \
           not all(isinstance(i, Symbol) for i in vars):
            return super()._print_Derivative(expr)

        # If you want the printer to work correctly for nested
        # expressions then use self._print() instead of str() or latex().
        # See the example of nested modulo below in the custom printing
        # method section.
        return "{}_{{{}}}".format(
            self._print(Symbol(function.func.__name__)),
                        ''.join(self._print(i) for i in vars))

    def _print_Max(self, expr):
        f = expr.func.__name__
        objective = expr.args[0]
        wrt = expr.args[1:]
        wrt_strings = []
        for w in wrt:
            wrt_strings.append(self._print(w[0]) + '\in' + self._print(w[1]))

        return "{}_{{{}}}({})".format(self._print(f),','.join(wrt_strings), self._print(objective))



def print_latex(expr):
    """ Most of the printers define their own wrappers for print().
    These wrappers usually take printer settings. Our printer does not have
    any settings.
    """
    return LaTeXPrinter().doprint(expr)


class CStrPrinter(StrPrinter):
    def _print_Parameter(self, expr):
        return expr._scalar.name


def print_str(expr):
    return CStrPrinter().doprint(expr)
