from core.agent import AbstractAgent, LinkedAgent
from core.model import Model
from core.market import Balances

from sympy import Eq, Function, Symbol, symbols, Expr, Integral, Derivative, exp, ln,\
    GreaterThan, LessThan


def example1():
    t = Symbol('t')
    x = Function('x')
    c = Function('c')
    r, delta, T, sigma, J = symbols(r'r \delta T \sigma J')

    objective = Eq(J, 1/T * Integral(exp(-delta*t) * ln(c(t)), (t, 0, T)))

    d1 = Eq(Derivative(x(t), t), r * (x(t) - c(t)))

    ineq1 = LessThan(sigma * c(t), x(t))
    ineq2 = GreaterThan(c(t), 0)

    a = AbstractAgent(
        name='Ramsay',
        objectives=[objective],
        equations=[d1],
        inequations=[ineq1, ineq2],
        params=[t, r, delta, T, sigma, J],
        functions=[x(t), c(t)]
      )
    a.process(skip_validation=True)

    print(a.Lagrangian)



def example2():
    # Two agents model

    # customer agent
    t, beta, delta, gamma, T, J = symbols(r't \beta \delta \gamma T J')
    r_l, p_y, pi_o, a_S = symbols(r'r_{l} p_{y} \pi_{o} a_{S}', cls=Function)
    C, N, S = symbols(r'C N S', cls=Function)

    objective = Eq(J, Integral(1/(1-beta) * C(t)**(1-beta) * exp(-delta*t), (t, 0, T)))
    d1 = Eq(Derivative(N(t), t), pi_o(t) - Derivative(S(t), t) + r_l(t) * S(t) - p_y(t) * C(t))

    ineq1 = GreaterThan(N(t), 0, eval=False)
    ineq2 = GreaterThan(N(T) + a_S(T) * S(T), gamma * (N(0) + a_S(0) * S(0)))

    H = AbstractAgent(
        name='H',
        objectives=[objective],
        equations=[d1],
        inequations=[ineq1, ineq2],
        params=[t, beta, delta, gamma, T, J],
        functions=[r_l(t), p_y(t), pi_o(t), a_S(t), C(t), N(t), S(t)]
    )
    H = LinkedAgent.from_abstract(H)
    H.process(skip_validation=True)

    # producer agent
    t, beta, delta, gamma, T, Q, b = symbols(r't \beta \delta \gamma T Q b')
    r_l, p_y, a_L, a_K = symbols(r'r_{l} p_{y} a_{L} a_{K}', cls=Function)
    Y, K, J, N, L, pi = symbols(r'Y K J N L \pi', cls=Function)

    objective = 1
    d1 = 2
    d2 = 3




if __name__ == '__main__':
    example2()