""" Abstract errors and warnings for ECOMOD"""


class Error(Exception):
    header = ""

    def __init__(self, **kwargs):
        super().__init__(self.header.format(**kwargs))


class Warn(Warning):
    header = ""

    def __init__(self, **kwargs):
        super().__init__(self.header.format(**kwargs))


"""# File contains text level errors for ECOMOD"""


class NonLaTeXableError(Error):
    header = "This {file} file does not contain \\begin{document} and assumed to be not latexable.\n" \
             "Please write your model in .tex file correctly"


class NonYAMLableError(Error):
    header = "Seems like you have not formatted model in yaml notation correctly.\n" \
             "Here's error: {err}"


class NonSympyfiableError(Error):
    header = "There is unprocessable sympy entity {err}"


class VariableAmbiguity(Error):
    header = "There is two variables matches the same thing {var1} and {var2}"


class ExtraVariableError(Error):
    header = "There are extra variables in model {vars}"


class TimeVariableNotFound(Error):
    header = "There is no variable which express Time in model"


class ObjectiveFunctionNotFound(Error):
    header = "There is no expression which realize objective function in model.\n" \
             " Please check this substring: '-->[extr,max,min]' to be in input"


class AnyPropertyNotFound(Error):
    header = "We cannot find this in your model : {attr}"


class DimensionInExpression(Error):
    header = "Please remove dimension for this expression: {expr} in input file."


class DimensionCheckingFailed(Error):
    header = "Please check these expression to be right in terms of dimension.\n" \
             "There are: {expr}"


class NoSuchFlow(Warn):
    header = "There is no such flow {flow} in this agent {agent}"


class NotRendered(Warn):
    header = "This model is not rendered: {model}"


class NotSubscriptedBalance(Error):
    header = 'Not all variables are tagged by agent tag. Agent tags {agent_names}'
