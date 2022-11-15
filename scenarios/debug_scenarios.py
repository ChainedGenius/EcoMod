from core.agent import LinkedAgent, create_empty_agent
from core.market import Flow
from core.model import Model
from core.utils import timeit


@timeit
def simple_linked_agents():
    f = '../inputs/agent.tex'
    A = LinkedAgent.read_from_tex(f)
    A.process()
    B = LinkedAgent.from_abstract(create_empty_agent('B'))
    C = LinkedAgent.from_abstract(create_empty_agent('C'))
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
    a3 = LinkedAgent.from_abstract(create_empty_agent('agent3'))
    a4 = LinkedAgent.from_abstract(create_empty_agent('agent4'))
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
    print(H.controls)

    f2 = '../models/inputs/Pmodel/P.tex'
    P = LinkedAgent.read_from_tex(f2)
    P.process(skip_validation=True)
    print(P.controls)
    print(P.phases)

@timeit
def p_model_dump():
    f1 = '../models/inputs/Pmodel/H.tex'
    H = LinkedAgent.read_from_tex(f1)
    H.process(skip_validation=True)

    f2 = '../models/inputs/Pmodel/P.tex'
    P = LinkedAgent.read_from_tex(f2)
    P.process(skip_validation=True)

    M = Model('Pmodel', [], [H, P])
    M.process()
    M.dump('../models/outputs/Pmodel')



if __name__ == "__main__":
    p_model_dump()
