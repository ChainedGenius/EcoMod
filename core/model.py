from pathlib import Path
from typing import List

from core.logger import log
from core.market import MarketValidator, Flow
from core.agent import AbstractAgent, AgentValidator, LinkedAgent, create_empty_agents
from templates.pprint import ModelTemplateEngine, exec_tex
from core.utils import timeit


class AgentMerger(object):

    def __find_ambiguity(self, lagents: List[AbstractAgent]):
        """
        Ambiguity == same variable name, but wrong dimension
        :return: True if found ambiguity, else False
        """
        pass

    def __rename_agent_local_variables(self, lagents: List[AbstractAgent]):
        pass

    def __other_merge_routine(self):
        "other merge routine"
        pass

    def merge(self, lagents: List[AbstractAgent]):
        self.__find_ambiguity(lagents)
        self.__other_merge_routine()
        self.__rename_agent_local_variables(lagents)


class ModelValidator(MarketValidator, AgentValidator, AgentMerger):
    def validate(self, markets, lagents):
        self.validate_agents(lagents)
        self.validate_market(markets)
        self.merge(lagents)


class Model(ModelValidator):
    """
        Merged model with several agents and markets
        or
        agents and balances
        or
        agents and flows
    """

    def __init__(self, name, balances=None, lagents: List[LinkedAgent] = None):
        self.balances = balances
        self.lagents = lagents
        self.name = name
        self.markets = []

    def add_agent(self, lagent: LinkedAgent):
        if lagent not in self.lagents:
            self.lagents.append(lagent)

    def process(self):
        pass
        #self.validate(self.markets, self.lagents)

    def dump(self, destination=None):
        if not destination:
            destination = '.'
        engine = ModelTemplateEngine()
        data = {}
        for agent in self.lagents:
            data[agent.name] = agent.compress(to_tex=True, headers=False)
        template_data = {"DATA": data, "BALANCES": self.balances.compress()["BALANCES"]}
        engine.render(template_data)
        tex_directorypath = Path(destination) / self.name
        tex_filepath = (tex_directorypath / self.name).with_suffix('.tex')
        tex_directorypath = Path(str(tex_directorypath.cwd()) + str(tex_directorypath))
        tex_filepath = Path(str(tex_filepath.cwd()) + str(tex_filepath))
        engine.dump(tex_filepath)
        exec_tex(tex_filepath, tex_directorypath)

    @log(comment="Visualizing...")
    def visualize(self, f):
        from hypernetx import Hypergraph
        from itertools import chain
        from hypernetx.drawing.rubber_band import draw
        import matplotlib.pyplot as plt
        isolated = {f'isolated {idx}': {i.name} for idx, i in enumerate(self.lagents) if not i.flows}
        flows = chain(*[a.flows for a in self.lagents])
        edges = {f'{l.value} ({l.dim})': {l.producer.name, l.receiver.name} for l in flows}
        edges.update(isolated)
        hg = Hypergraph(edges)
        draw(hg)
        if f:
            plt.savefig(f)


if __name__ == "__main__":
    f1 = '../inputs/agent.tex'
    f2 = '../inputs/agent2.tex'
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
    m.visualize()
