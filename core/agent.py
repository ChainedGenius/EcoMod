from deserialiser import read_model_from_tex
from sympyfier import sympify, ecomodify

class Agent(object):
    def __init__(self, inequations=None, equations=None, functions=None, params=None):
        if inequations is None:
            inequations = []
        if equations is None:
            equations = []
        if functions is None:
            functions = []
        if params is None:
            params = []
        self.params = params
        self.functions = functions
        self.equations = equations
        self.inequations = inequations

    def read_from_tex(self, f):
        header, raw_model = read_model_from_tex(f)
        model = ecomodify(raw_model)
        return Agent(*model)


if __name__ == "__main__":
    f = 'inputs/agent.tex'
    A = Agent().read_from_tex(f)
    print(A.__class__)
    print(A.equations)
