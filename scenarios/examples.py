from core.agent import AbstractAgent, LinkedAgent
from core.model import Model
from core.market import Balances

from sympy import Eq, Function, Symbol, symbols, Expr, Integral, Derivative, exp, ln, \
    GreaterThan, LessThan

from utils import get_root_dir

project_path = get_root_dir()

def example1():
    t = Symbol('t')
    x = Function('x')
    c = Function('c')
    r, delta, T, sigma, J = symbols(r'r \delta T \sigma J')

    objective = Eq(J, 1 / T * Integral(exp(-delta * t) * ln(c(t)), (t, 0, T)))

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

    objective = Eq(J, Integral(1 / (1 - beta) * C(t) ** (1 - beta) * exp(-delta * t), (t, 0, T)))
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

    objective = Eq(Q, Integral(1 / (1 - beta) * (pi(t) / p_y(t)) ** (1 - beta) * exp(-delta * t), (t, 0, T)))
    d1 = Eq(Derivative(N(t), t), p_y(t) * J(t) + Derivative(L(t), t) - r_l(t) * L(t) - pi(t))
    d2 = Eq(Derivative(K(t), t), J(t))
    b1 = Eq(Y(t), 1 / b * K(t))

    ineq1 = GreaterThan(N(t), 0)
    ineq2 = GreaterThan(a_L(T) * L(T) + a_K(T) * K(T), gamma * (N(0) + a_L(0) * L(0) + a_K(0) * K(0)))

    P = AbstractAgent(
        name='P',
        objectives=[objective],
        equations=[d1, d2, b1],
        inequations=[ineq1, ineq2],
        params=[t, beta, delta, gamma, T, Q, b],
        functions=[r_l(t), p_y(t), a_L(t), a_K(t), Y(t), K(t), J(t), N(t), L(t), pi(t)]
    )
    P = LinkedAgent.from_abstract(P)
    P.process(skip_validation=True)

    # flows and balances
    Y_P, C_H, J_P, L_P, S_H, N_P, N_H, pi_P, pi_o = symbols(r'Y_P C_H J_P L_P S_H N_P N_H pi_P pi_o', cls=Function)
    flow1 = Eq(Y_P(t), C_H(t) + J_P(t))
    flow2 = Eq(L_P(t), S_H(t))
    flow3 = Eq(N_P(t), -N_H(t))
    flow4 = Eq(pi_P(t), pi_o(t))
    B = Balances(['H', 'P'], [flow1, flow2, flow3, flow4])

    # final model
    M = Model('Pmodel', B, [H, P])
    M.process()
    print(M.lagents[0].Lagrangian)
    print(M.lagents[1].Lagrangian)


def example3():
    # Parameters
    t, rho, delta, eps1, eps2, T = symbols(r't \rho \delta \epsilon_1 \epsilon_2 T')
    B, B1, R, R_s, R_l, M, eta, L0, S0, W0 = symbols(r'B B_1 R R_s R_l M \eta L_0 S_0 W_0')
    C1, C2, C3 = symbols(r'C_1 C_2 C_3')
    J = symbols('J')
    # Functions
    x1, x2, x3, u1, u2, u3 = symbols('x_1 x_2 x_3 u_1 u_2 u_3', cls=Function)

    # Functional
    obj = Eq(J, Integral(
        -(u3(t)) ** (1 - rho) * exp(-delta * t) - eps1 * (u2(t)) ** 2 - eps2 * (u1(t)) ** 2,
        (t, 0, T)))

    # Diffeq
    diffeq1 = Eq(
        x1(t).diff(t),
        B1 * (x1(t) * u1(t) + x2(t) * u2(t) + x3(t) * u3(t)) + B * x1(t) + R
    )
    diffeq2 = Eq(
        x2(t).diff(t),
        B1 * (x1(t) * u1(t) + x2(t) * u2(t) + x3(t) * u3(t)) + B * x2(t) + R
    )
    diffeq3 = Eq(
        x3(t).diff(t),
        B1 * (x1(t) * u1(t) + x2(t) * u2(t) + x3(t) * u3(t)) + B * x3(t) + R
    )

    bd1 = LessThan(C1 * x1(t) + C2 * x2(t) + C3 * x3(t), 0)
    bd2s = [
        GreaterThan(u1(t), 0), LessThan(u1(t), R_s),
        GreaterThan(u2(t), 0), LessThan(u2(t), R_l),
        GreaterThan(u3(t), 0), LessThan(u3(t), M),
        GreaterThan(eta * x3(0) - x3(T), 0)
    ]
    bd3s = [
        Eq(x1(0), S0),
        Eq(x2(0), L0),
        Eq(x3(0), W0)  # ,
        # Eq(T, 3)
    ]

    P = AbstractAgent(
        name='A',
        objectives=[obj],
        equations=[diffeq1, diffeq2, diffeq3, *bd3s],
        inequations=[bd1, *bd2s],
        params=[
            rho, delta, eps1, eps2, T,
            B, B1, R, R_s, R_l, M, eta,
            L0, S0, W0, C1, C2, C3, J
        ],
        functions=[x1(t), x2(t), x3(t), u1(t), u2(t), u3(t)]
    )
    P = LinkedAgent.from_abstract(P)
    P.process(skip_validation=True)
    P.dump(project_path + '/models/outputs/CZF/', is_absolute=True)


if __name__ == '__main__':
    example3()
