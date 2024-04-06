from sympy import symbols, Function, Integral, Eq

from task_classifier.ecomodify import classify_task
from serializers.latex import LaTeXSerializer

from task_classifier.classifier import TaskSanitizer
from utils import MAX, make_raw_dict


def variational():
    t, A, B = symbols('t A B')
    x = Function('x')
    eqs = [Eq(MAX, Integral(A * (x(t).diff(t)) ** 2 + B * t * x(t), (t, 0, 1))),
           Eq(x(0), 0),
           Eq(x(1), 0)]


    raw_dict = TaskSanitizer(make_raw_dict(eqs + [x(t), t, A, B]))
    print(classify_task(**raw_dict.kwargs).__class__)


def opt_control1_from_latex():
    a = LaTeXSerializer.serialize(r'C:\Users\BaTe\Documents\GitHub\EcoMod\models\inputs\Pmodel_2products\H.tex')
    a = TaskSanitizer(a)
    print(a.kwargs)
    print(classify_task(**a.kwargs).__class__)

if __name__ == '__main__':
    opt_control1_from_latex()