One-agent (isolated) models
============================
.. autoclass:: core.agent.AbstractAgent

.. code:: python

    from core.agent import AbstractAgent
    from sympy import symbols, Function, Eq, Integral, Derivative, LessThan, GreaterThan, ln, exp
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




Multi-agent models
============================

.. code:: python

    from core.agent import AbstractAgent, LinkedAgent
    from core.model import Model
    from market import Balances
    from sympy import symbols, Function, Eq, Integral, Derivative, LessThan, GreaterThan, ln, exp

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
