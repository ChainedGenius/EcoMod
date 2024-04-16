from core.deserialiser import read_tex, read_model_from_tex
from core.ecomod_utils import is_substricted, remove_subscript, latexify
from core.errors.RWErrors import NotSubscriptedBalance
from core.sympyfier import ecomodify, ecomodify_


class Flow(object):
    """
        Here we will contain parsing results from flows file?
    """

    def __init__(self, producer, receiver, value, dim):
        self.producer = producer
        self.receiver = receiver
        self.value = value
        self.dim = dim
        pass

    def __str__(self):
        return f'{self.producer.name} ---- {self.value} ---> {self.receiver.name}'

    def invert(self, new=False):
        if new:
            return Flow(self.receiver, self.producer, self.value, self.dim)
        else:
            buf = [self.producer, self.receiver]
            self.producer = buf[1]
            self.receiver = buf[0]


class Market(object):
    def __init__(self, eq, dim, lagents):
        self.eq = eq
        self.dim = dim
        self.lagents = lagents
        pass


class MarketValidator(object):
    def __market_closureness(self, markets):
        pass

    def validate_market(self, markets):
        self.__market_closureness(markets)


class Balances(object):
    def __init__(self, agent_names, eqs):
        self.balances = eqs
        self.agent_names = agent_names
        self.dependencies = {}

    def __only_tagged(self):
        """
        collect all atoms
        :return: raise errors
        """

        from numpy import prod

        # collect atoms

        ret = prod([is_substricted(a) for a in self.atoms()])
        if ret != 1:
            raise NotSubscriptedBalance(agent_names=self.agent_names)

    def validate(self):
        self.__only_tagged()

    def __collect_involved_agents(self):
        ret = {}
        for name in self.agent_names:
            ret[name] = [b for b in self.balances if f"_{{{name}}}" in b.__str__()]

        self.dependencies = ret

    def atoms(self, name=None, detag=True):
        from itertools import chain
        from sympy import Symbol, Function
        if not name:
            return set(chain(*[eq.atoms(Symbol, Function) for eq in self.balances]))
        ret = set(
            chain(*[{a for a in eq.atoms(Symbol, Function) if is_substricted(a, tag=name)} for eq in self.balances]))
        if not detag:
            return ret
        return {remove_subscript(a) for a in ret}

    @classmethod
    def read_from_tex(cls, f):
        header, content = read_model_from_tex(f)
        _, _, equations, _, params, _ = ecomodify_(content)
        agents_tags = [p.name for p in params]
        balances = equations
        return cls(agents_tags, balances)

    def process(self):
        # self.validate()
        self.__collect_involved_agents()

    def compress(self):
        return {
            "BALANCES": latexify(self.balances)
        }


if __name__ == "__main__":
    f = '../models/inputs/Pmodel/flows.tex'
    b = Balances.read_from_tex(f)
    b.process()
    print(b.dependencies)
    print(b.atoms(name='H'))
    print(b.atoms(name='P'))
