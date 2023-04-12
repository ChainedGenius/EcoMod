from core.agent import LinkedAgent, create_empty_agents, AbstractAgent
from core.market import Flow, Market, Balances
from core.model import Model
from core.utils import timeit


@timeit
def simple_agent():
    from sympy import Function
    from itertools import chain
    f = '../models/inputs/Pmodel/H.tex'
    A = LinkedAgent.read_from_tex(f)
    A.process(skip_validation=True)
    print(A.lagrangian)
    print(A.Lagrangian)
    print(A.phases)
    print(A.transversality_conditions())
    A.dump('test/')


@timeit
def simple_linked_agents():
    f = '../inputs/agent.tex'
    A = LinkedAgent.read_from_tex(f)
    A.process()
    B = LinkedAgent.from_abstract(create_empty_agents('B'))
    C = LinkedAgent.from_abstract(create_empty_agents('C'))
    A.add_flow(Flow(A, C, 3, 'rub'))
    B.add_flow(Flow(A, B, 6, 'tv'))
    print(A.__class__())
    print(A.name)
    print(A.Lagrangian)
    print(A.lagrangian)
    print(A.euler_equations())
    print(A.transversality_conditions())
    print(A.control_optimality())
    print(A.KKT())
    print(A.print_flows())


@timeit
def simple_model_viz():
    f1 = '../models/inputs/agent.tex'
    f2 = '../models/inputs/agent2.tex'
    a1 = LinkedAgent.read_from_tex(f1)
    a1.process()
    a2 = LinkedAgent.read_from_tex(f2)
    a3 = LinkedAgent.from_abstract(create_empty_agents('agent3'))
    a4 = LinkedAgent.from_abstract(create_empty_agents('agent4'))
    flow1 = Flow(a1, a2, 50, 'rub')
    flow2 = Flow(a2, a1, 1, 'tv')
    a1.add_flow(flow1)
    a2.add_flow(flow2)
    m = Model('1', [], [a1, a2, a3, a4])
    m.visualize('test.png')


@timeit
def agent_dump():
    # broken scenario
    """
        raise TemplateNotFound(template)
        jinja2.exceptions.TemplateNotFound: LAgent.tex

    """
    f1 = '../models/inputs/agent.tex'
    A = LinkedAgent.read_from_tex(f1)
    A.process()
    A.dump('../models/outputs/Amodel')


@timeit
def p_model():
    f1 = '../models/inputs/Pmodel/H.tex'
    H = LinkedAgent.read_from_tex(f1)
    H.process(skip_validation=True)
    print(H.__dict__)

    f2 = '../models/inputs/Pmodel/P.tex'
    P = LinkedAgent.read_from_tex(f2)
    P.process(skip_validation=True)
    print(P.__dict__)


@timeit
def p_model_dump():
    f1 = '../models/inputs/Pmodel/H.tex'
    H = LinkedAgent.read_from_tex(f1)
    H.process()

    f2 = '../models/inputs/Pmodel/P.tex'
    P = LinkedAgent.read_from_tex(f2)
    P.process()

    B = Balances.read_from_tex('../models/inputs/Pmodel/flows.tex')
    M = Model('Pmodel', B, [H, P])
    M.process()
    M.dump('../models/outputs/Pmodel')


@timeit
def model_viz():
    # no real ecomod models
    # only viz scenario
    A, B, C, D, E = create_empty_agents('A B C D E', cls=LinkedAgent)
    f1 = Flow(A, C, 1, '')
    f2 = Flow(C, D, 2, '')
    f3 = Flow(C, E, 3, '')
    f4 = Flow(D, E, 4, '')
    A.add_flow(f1)
    C.add_flow(f2)
    C.add_flow(f3)
    D.add_flow(f4)
    M = Model('viz', [], [A, B, C, D, E])
    M.visualize('../models/outputs/Pmodel')


@timeit
def pgmodel():
    f1 = '../models/inputs/Pmodel/H.tex'
    H = LinkedAgent.read_from_tex(f1)
    H.process(skip_validation=True)

    f2 = '../models/inputs/Pmodel/P.tex'
    P = LinkedAgent.read_from_tex(f2)
    P.process(skip_validation=True)
    G = create_empty_agents('G', cls=LinkedAgent)

    B = Balances.read_from_tex('../models/inputs/Pmodel/flows.tex')
    M = Model('Pmodel', B, [H, P])
    M.process()
    M.dump('../models/outputs/Pmodel')


@timeit
def p2model():
    from sympy.printing.latex import latex
    f1 = '../models/inputs/Pmodel_2products/H.tex'
    H = AbstractAgent.read_from_tex(f1)
    H.process(skip_validation=True)

    f2 = '../models/inputs/Pmodel_2products/P.tex'
    P = AbstractAgent.read_from_tex(f2)
    P.process(skip_validation=True)
    B = Balances.read_from_tex('../models/inputs/Pmodel_2products/flows.tex')

    M = Model('Pmodel', B, [H, P])
    M.process()
    M.dump('../models/outputs/Pmodel_2products')





if __name__ == "__main__":
    simple_agent()
    # xrepalce error: something wrong with objective functions and xreplace=False in agent parsing
