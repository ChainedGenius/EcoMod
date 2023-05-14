One-agent (isolated) models
============================

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

