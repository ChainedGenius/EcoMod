from core.agent import AbstractAgent, LinkedAgent
from core.model import Model
from core.market import Balances

from sympy import *

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


def example_3_star():
    t, T, rho, delta, eps1, eps2, \
        alpha_l, gamma_l, beta_l, \
        alpha_s, gamma_s, beta_s, \
        OC, n_s = symbols(r't T \rho \delta \epsilon_1 \epsilon_2 \alpha_l \gamma_l \beta_l \alpha_s \gamma_s '
                          r'\beta_s OC n_s')

    R_l, R_s, M, tau_s, tau_l, w_l, K_A = \
        symbols(r'R_l R_s M \tau_s \tau_l w_l K_A')

    L_0, S_0, A_0, eta, J = symbols(r'L_0 S_0 A_0 \eta J')

    Z, r_l, r_s, L, S, A = symbols('Z r_l r_s L S A' , cls=Function)

    objective = Eq(J, Integral(Z(t) ** (1-rho) * exp(-delta * t) - eps1 * (r_l(t))**2 - eps2 * (r_s(t))**2, (t, 0, T)))
    diffeq1 = Eq(diff(L(t), t), L(t) * (alpha_l - gamma_l * r_l(t) - beta_l))
    diffeq2 = Eq(diff(S(t), t), S(t) * (alpha_s + gamma_s * r_l(t) - beta_s))
    diffeq3 = Eq(
        diff(A(t), t),
        -OC - Z(t) +
        L(t) * ((gamma_l + 1) * r_l(t) + beta_l - alpha_l) +
        S(t) * ((-1 + (1 - n_s) * gamma_s) * r_s(t) + (1 - n_s) * (alpha_s * beta_s))
    )

    eqs = [
        Eq(L(0), L_0),
        Eq(A(0), A_0),
        Eq(S(0), S_0),
    ]

    ineqs = [
        GreaterThan(r_l(t), 0),
        LessThan(r_l(t), R_l),
        GreaterThan(r_s(t), 0),
        LessThan(r_s(t), R_s),
        GreaterThan(Z(t), 0),
        LessThan(Z(t), M),
        GreaterThan(A(t) - tau_s * S(t) - tau_l * L(t), 0),
        GreaterThan(L(t) * (1 * K_A * w_l) + (1 - K_A) * A(t) - S(t)*(1 - n_s), 0),
        LessThan(A(0) + L(0) - (1 - n_s) * S(0), eta * (A(T) + L(T) - (1 - n_s) * S(T)))
    ]

    P = AbstractAgent(
        name='A',
        objectives=[objective],
        equations=[diffeq1, diffeq2, diffeq3, *eqs],
        inequations=ineqs,
        params=[
            T, rho, delta, eps1, eps2,
            alpha_l, gamma_l, beta_l,
            alpha_s, gamma_s, beta_s,
            OC, n_s, R_l, R_s, M, tau_s,
            tau_l, w_l, K_A, L_0, S_0, A_0, eta
        ],
        functions=[Z(t), r_l(t), r_s(t), L(t), S(t), A(t)]
    )
    P = LinkedAgent.from_abstract(P)
    P.process(skip_validation=True)
    # P.dump(project_path + '/models/outputs/AL_bank/', is_absolute=True)
    print(P.regularity_conditions())


def example3_matrix_form():
    # Parameters
    t, rho, delta, eps1, eps2, T = symbols(r't \rho \delta \epsilon_1 \epsilon_2 T')
    B, B1, R, R_s, R_l, M, eta, L0, S0, W0 = symbols(r'B B_1 R R_s R_l M \eta L_0 S_0 W_0')
    C1, C2, C3 = symbols(r'C_1 C_2 C_3')
    B11, B12, B13 = symbols(r'B_{11} B_{12} B_{13}')
    J = symbols('J')
    # Functions
    x1, x2, x3, u1, u2, u3 = symbols('x_1 x_2 x_3 u_1 u_2 u_3', cls=Function)

    # Vectors and Matrices
    X = Matrix([[x1(t), x2(t), x3(t)]]).T
    U = Matrix([[u1(t), u2(t), u3(t)]]).T
    C = MatrixSymbol('C', 2, 3).T
    B1_m = MatrixSymbol('B_1', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    R = MatrixSymbol('R', 3, 1)
    # Objective
    obj = Eq(J, Integral(
        -(u3(t)) ** (1 - rho) * exp(-delta * t) - eps1 * (u2(t)) ** 2 - eps2 * (u1(t)) ** 2,
        (t, 0, T)))

    # diffeq <vector mode>

    diffeq = Eq(X.diff(t), B1_m * X)  # * U + B * X + R)

    pprint(diffeq)


if __name__ == '__main__':
    example3()
