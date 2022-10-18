from deserialiser import read_model_from_tex
from sympyfier import sympify, ecomodify

class Agent(object):
    def __init__(self):
        pass

    def read_from_tex(self, f):
        header, raw_model = read_model_from_tex(f)
        model = ecomodify(raw_model)
        print(model)


if __name__ == "__main__":
    f = 'inputs/agent.tex'
    a = Agent()
    a.read_from_tex(f)