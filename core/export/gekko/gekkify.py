from core.agent import LinkedAgent
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO


def guide_case():
    import numpy as np
    import matplotlib.pyplot as plt
    from gekko import GEKKO

    m = GEKKO()

    nt = 101
    m.time = np.linspace(0, 2, nt)

    # Variables
    x1 = m.Var(value=1)
    x2 = m.Var(value=0)
    u = m.Var(value=0, lb=-1, ub=1)

    p = np.zeros(nt)
    p[-1] = 1.0
    final = m.Param(value=p)

    # Equations
    m.Equation(x1.dt() == u)
    m.Equation(x2.dt() == 0.5 * x1 ** 2)

    # Objective Function
    m.Obj(x2 * final)

    m.options.IMODE = 6
    m.solve()
    # m.solve(remote=False)  # for local solve

    plt.figure(1)
    plt.plot(m.time, x1.value, 'k-', lw=2, label=r'$x_1$')
    plt.plot(m.time, x2.value, 'b-', lw=2, label=r'$x_2$')
    plt.plot(m.time, u.value, 'r--', lw=2, label=r'$u$')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    pass


class Gekkifier(object):

    def __init__(self):
        self.model = GEKKO()

    def cast_emodel_params(self, emodel):
        exclude_list = [i.lhs for i in emodel.objectives] + [emodel.time]

        params = set(emodel.params) - set(exclude_list)
        print([type(i) for i in emodel.params])
        print([type(i) for i in exclude_list])
        return list(params)

    def convert(self):
        return self.model

    def __gather_boundaries(self, funcs, boundaries):
        from sympy import Function

        ret = {}
        for b in boundaries:
            ret[b] = b.find(Function)


        for f in funcs:
            ret[f] = [i for i in boundaries if f in i.find(Function) ]

        return ret



    def convert_funcs(self, ft: dict):
        base_guess = 0
        for k, v in ft:
            # k -- name of variable, v -- value for cauchy task
            self.model.Var(value=v or base_guess, name=k)


if __name__ == '__main__':
    m = GEKKO()
    c = Gekkifier()
    f = '../../../models/inputs/agent.tex'
    a = LinkedAgent.read_from_tex(f)
    a.process(skip_validation=True)
    print(c.cast_emodel_params(a))
    print(a.objectives)







    pass
