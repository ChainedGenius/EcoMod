==============
Support engine
==============

----------
Validation
----------

-----------
AgentMerger
-----------

.. autofunction:: core.model.AgentMerger

--------------
ModelValidator
--------------

.. code:: python
    class ModelValidator(MarketValidator, AgentValidator, AgentMerger):
    def validate(self, markets, lagents):
        self.validate_agents(lagents)
        self.validate_market(markets)
        self.merge(lagents)

-----
Model
-----

.. code:: python
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

------
Errors
------
.. code:: python
    class NonLaTeXableError(Error):
        header = "This {file} file does not contain \\begin{document} and assumed to be not latexable.\n" \
                 "Please write your model in .tex file correctly"

.. code:: python
    class NonYAMLableError(Error):
        header = "Seems like you have not formatted model in yaml notation correctly.\n" \
                 "Here's error: {err}"

.. code:: python
    class NonSympyfiableError(Error):
        header = "There is unprocessable sympy entity {err}"

.. code::python
    class VariableAmbiguity(Error):
        header = "There is two variables matches the same thing {var1} and {var2}"

.. code:: python
    class ExtraVariableError(Error):
        header = "There are extra variables in model {vars}"

.. code:: python
    class TimeVariableNotFound(Error):
        header = "There is no variable which express Time in model"

.. code:: python
    class ObjectiveFunctionNotFound(Error):
        header = "There is no expression which realize objective function in model.\n" \
                 " Please check this substring: '-->[extr,max,min]' to be in input"

.. code:: python
    class AnyPropertyNotFound(Error):
        header = "We cannot find this in your model : {attr}"

.. code:: python
    class DimensionInExpression(Error):
        header = "Please remove dimension for this expression: {expr} in input file."

.. code:: python
    class DimensionCheckingFailed(Error):
        header = "Please check these expression to be right in terms of dimension.\n" \
                 "There are: {expr}"

.. code:: python
    class NoSuchFlow(Warn):
        header = "There is no such flow {flow} in this agent {agent}"

.. code:: python
    class NotRendered(Warn):
        header = "This model is not rendered: {model}"

.. code:: python
    class NotSubscriptedBalance(Error):
        header = 'Not all variables are tagged by agent tag. Agent tags {agent_names}'