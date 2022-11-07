from agent import Agent
from typing import List

from utils import timeit


class Flow(object):
    """
        Here we will contain parsing results from flows file?
    """

    def __init__(self, producer, receiver, value):
        self.producer = producer
        self.receiver = receiver
        self.value = value
        pass


class Market(object):
    def __init__(self, eq, dim, lagents):
        self.eq = eq
        self.dim = dim
        self.lagents = lagents
        pass


class LinkedAgent(Agent):
    def __init__(self, flows: List[Flow]):
        super().__init__()
        self.flows = flows


class MarketValidator(object):
    def __market_closureness(self, markets):
        pass

    def validate_market(self, markets):
        self.__market_closureness(markets)


class LAgentValidator(object):
    # for links mb redirected to Lagents class
    def __variable_check(self, agents: List[Agent]):
        pass

    def __dimension_check(self, agents: List[Agent]):
        pass

    # isolated model checks
    def __variable_completeness(self, agents: List[Agent]):
        pass

    def validate_agents(self, agents: List[Agent]):
        self.__variable_check(agents)
        self.__dimension_check(agents)
        self.__variable_completeness(agents)


class AgentMerger(object):

    def __find_ambiguity(self, lagents: List[Agent]):
        """
        Ambiguity == same variable name, but wrong dimension
        :return: True if found ambiguity, else False
        """
        pass

    def __rename_agent_local_variables(self, lagents: List[Agent]):
        pass

    def __other_merge_routine(self):
        "other merge routine"
        pass

    def merge(self, lagents: List[Agent]):
        self.__find_ambiguity(lagents)
        self.__other_merge_routine()
        self.__rename_agent_local_variables(lagents)



class ModelValidator(MarketValidator, LAgentValidator, AgentMerger):
    def validate(self, markets, lagents):
        self.validate_agents(lagents)
        self.validate_market(markets)
        self.merge(lagents)


class Model(ModelValidator):
    """
        Merged model with several agents and markets
    """

    def __init__(self, markets, lagents):
        self.markets = markets
        self.lagents = lagents
        pass

    def process(self):
        self.validate(self.markets, self.lagents)

    @timeit
    def visualize(self):
        from hypernetx import Hypergraph
        from itertools import chain
        from hypernetx.drawing.rubber_band import draw
        agent_names = [a.name for a in self.lagents]
        flows = chain([a.flows for a in self.lagents])
        edges = [{l.producer.name, l.receiver.name} for l in flows]
        hg = Hypergraph(edges)
        draw(hg)


if __name__ == "__main__":
    from hypernetx import Hypergraph
    from itertools import chain
    from hypernetx.drawing.rubber_band import draw
    import matplotlib.pyplot as plt
    f1 = 'inputs/agent.tex'
    A1 = Agent().read_from_tex(f1)
    A1.process(skip_validation=True)
    f2 = 'inputs/agent2.tex'
    A2 = Agent().read_from_tex(f2)
    A2.process(skip_validation=True)
    hg = Hypergraph({0: (A1.name, A2.name)})
    fig = draw(hg)
    plt.show()
