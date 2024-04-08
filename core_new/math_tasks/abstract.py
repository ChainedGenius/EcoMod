from datamodel.config import Config

class SystemProblem:
    def __init__(self, arguments, equations, inequations, config=None):
        if config is None:
            config = Config()


