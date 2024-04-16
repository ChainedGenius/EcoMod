from sympy import symbols, Function, Integral, Eq

from task_classifier.ecomodify import classify_task
from serializers.latex import LaTeXSerializer

from task_classifier.classifier import TaskSanitizer
from utils import MAX, make_raw_dict, iterable_substract


def variational():
    t, A, B = symbols('t A B')
    x = Function('x')
    eqs = [Eq(MAX, Integral(A * (x(t).diff(t)) ** 2 + B * t * x(t), (t, 0, 1))),
           Eq(x(0), 0),
           Eq(x(1), 0)]

    raw_dict = TaskSanitizer(make_raw_dict(eqs + [x(t), t, A, B]))
    print(classify_task(**raw_dict.kwargs).__class__)


def opt_control1_from_latex():
    a, name = LaTeXSerializer.serialize(r'C:\Users\BaTe\Documents\GitHub\EcoMod\models\inputs\Pmodel_2products\H.tex')
    a = TaskSanitizer(a, name)
    print(a.functions)
    print(a.config_template)
    print(a.config)
    # print(k:=[*a.kwargs.get('config').keys()])
    # print(m:=[*a.kwargs.get('variable_info').keys()])
    # print(iterable_substract(k, m))
    # print(iterable_substract(m,k))
    task = classify_task(**a.kwargs)
    print(task.functions)
    print(task.__class__)
    print(task.Lagrangian)
    print(task.objectives)
    print(task.controls)

if __name__ == '__main__':
    opt_control1_from_latex()

