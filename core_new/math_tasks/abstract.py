class ConfigurableProblem:
    def __init__(self, config, config_template):
        self.config = config
        self.config_template = config_template
        self.__validate_cfg()
        pass

    def __validate_cfg(self):
        if not self.config.keys() == self.config_template.keys():
            raise KeyError(f'Config keys is not equal')

        if not [i.is_Constant for i in self.config.values()]:
            pass

class AnyProblem:
    def __init__(self, functions, equations, inequations, config=None, config_template=None):

        pass