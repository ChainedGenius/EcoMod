---------
.. code:: python
    class AbstractAgent(AgentValidator):
    """
        Core methods for Agent models.

        Methods:

            Constructors:

                1. __init__: Init Agent from its parts: see method args.
                2. read_from_tex: Parse Agent model from .tex file written in YAML (json-like) format. See examples at `/models/inputs/`.

            Private:

                1. __generate_duals: Generate dual variables and functions due to Lagrange principum.

            Public:

                Additional:

                    1. Support system

                    Methods that helps system to understand which variables are phase, which are controls etc.

                        1. time : extract time variable in agent model
                        2. time_horizon : extract time horizon boundaries from integral part of objective functions
                        3. transitions : extract first-order differential equations from all boundaries
                        4. phases : extract phase variables from all functions
                        5. external : extract exogenous constants from all functions
                        6. controls : extract control variables from all functions
                        7. variables : extract all model variable (not-functions)
                        8. expr : extract all expressions (with objectives)
                        9. boundaries : all equations and inequations (no differential equations)
                        10. constant_ineqs : inequations with no terminal values
                        11. diff_degree : Divide all expressions over differential equation degrees
                        12. validate : provide Agent model validation from subclass

                    2. Misc

                    Basic class methods to provide comfort.

                        1. args : Args to re-init
                        2. kwargs : Kwargs to re-init
                        3. process : Pre-process model to be used in core methods
                        4. compress : Evaluate all Pontragin Principle conditions
                        5. dump : Process model to PDF file with optimal conditions (Pontryagin Principle) [compress + dump]

                    3. Core

                    Methods that conduct Maximum Principle Conditions due to Lagrange principum.

                        1. Lagrangian : integral part of L
                        2. lagrangian : termination part of L
                        3. euler_equations : Euler-Lagrange equations
                        4. transversality_conditions : Transversality conditions
                        5. control_optimality : Control optimality conditions
                        6. KKT: Dual-feasibility and Complementary Slackness conditions


    """
    emitent = False

    def __init__(self, name='', objectives=None, inequations=None, equations=None, functions=None, params=None,
                 dim_dict=None):
        if dim_dict is None:
            dim_dict = {}
        if objectives is None:
            objectives = []
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
        self.objectives = objectives
        self.dim_dict = dim_dict
        self.name = name
        # dual variables k:v where k = real phase or ineq constraint, v = dual
        self.lambdas = {}  # for conjugate arbitrary constants
        self.duals = {}  # for conjugate functions
        # linking part)
        self.links = []

        self.processed = False

    @log(comment='Agent ready for analysis')
    def __generate_duals(self):
        """
        Lambdas -- dual variables, conducted by count of Len(chain(objectives, boundaries, constant_ineqs))
        Alphas -- dual functions, conducted by count of Len(chain(transitions, [inequations - constant_ineqs] + boundaries))
        :return: self.lambdas -- List[Symbol]
                 self.alphas -- List[Function]
        """
        from sympy import Symbol, Function
        # step 1: create lambdas
        lambda_factor = self.objectives + self.boundaries + self.constant_ineqs
        lambdas_count = len(lambda_factor)
        self.lambdas = {
            (eq2func(lambda_factor[i]) if lambda_factor[i] not in self.objectives else lambda_factor[i].args[1]): v for
            i, v in enumerate(generate_symbols(tag='lambda', count=lambdas_count, cls=Symbol))}
        # step 2: create conjugates funcs (duals)
        alpha_factors = self.transitions + iterable_substract(self.inequations, self.constant_ineqs) + self.boundaries
        alpha_count = len(alpha_factors)
        self.duals = {eq2func(alpha_factors[i]): d(self.time) for i, d in
                      enumerate(generate_symbols(tag='alpha', count=alpha_count, cls=Function))}

    @property
    def time(self):
        """
        Extract time variable from:
            1. Integral in objectives
            2. Differential variable in boundaries

        :return: Union[Symbol, TimeVariableNotFound]

        """
        # two ways: integration argument, under derivative
        from sympy import Integral
        # method 1
        try:
            ret1 = self.objectives[0].find(Integral).pop().args[1][0]
        except:
            ret1 = None
        # method 2
        try:
            from sympy import Derivative
            ret2 = [eq.find(Derivative).pop().args[1][0] for eq, deg in self.diff_degree(deg=1).items() if deg == 1][0]
        except:
            ret2 = None
        if not ret1 and not ret2:
            raise TimeVariableNotFound()
        return ret1 if ret1 is not None else ret2

    @property
    def time_horizon(self):
        """
        If time is extracted from objective integrals then collect integral boundaries.
        :return: Union[Tuple, AnyPropertyNotFound]
        """
        from sympy import Integral
        # method 1
        try:
            ret = self.objectives[0].find(Integral).pop().args[1][1:]
            return ret
        except:
            raise AnyPropertyNotFound(attr='Time Horizon')

    @property
    def transitions(self):
        """
        Extract linear differential equations from model.
        :return: List[Eq]
        """
        # linear Differential equations in model
        return [eq for eq in self.diff_degree(deg=1).keys()]

    @property
    def phases(self):
        """
        Extract phase variables from model: functions which appears as arguments in Differential operators in transitions
        :return: Union[List[Symbol], AnyPropertyNotFound]
        """
        from sympy import Derivative
        derivas = set(chain(*[d.find(Derivative) for d in [t for t in self.transitions]]))
        phases = [d.args[0] for d in derivas]
        indicator = prod([(p in self.functions) for p in phases])
        if indicator != 1:
            missing_var = iterable_substract(phases, self.functions)
            missing_transition = iterable_substract(self.functions, phases)
            if missing_var:
                raise AnyPropertyNotFound(attr=missing_var)
            if missing_transition:
                raise AnyPropertyNotFound(attr=missing_transition)
        return phases

    @property
    def external(self):
        """
        Extract exogenous functions, they must be substricted
        :return: List[Symbol]
        """
        return [ext for ext in self.functions if (is_substricted(ext) and not is_substricted(ext, tag=self.name))]

    @property
    def controls(self):
        """
        Extract all controls: controls = functions - external - phases
        :return: List[Symbol]
        """
        return iterable_substract(self.functions, self.phases + self.external)

    @property
    def expr(self):
        """
        All expressions in model
        :return: List[Expr]
        """
        return self.equations + self.inequations + self.objectives

    @property
    def variables(self):
        """
        All symbols in model
        :return: List[Symbol]
        """
        return self.functions + self.params

    @property
    def boundaries(self):
        """
        Expressions with these types:

            1. Differential degree == 0
            2. Constant inequalities

        :return: List[Expr]
        """
        # Equality types (deg=0) and constant ineqs
        return [*self.diff_degree(deg=0).keys()]

    @property
    def constant_ineqs(self):
        """
        Constant inequalities: those which contains only functions in terminant (constant) values.
        :return:
        """
        return [i for i in self.inequations if self.time not in i.free_symbols]

    @property
    def Lagrangian(self):
        """
        Intergal part of L
        :return: Expr
        """
        from sympy import Integral
        return span_dict(self.duals) + span_dict(
            {k: v for k, v in self.lambdas.items() if k in [i.args[1] for i in self.objectives]}).replace(
            lambda x: isinstance(x, Integral), lambda x: x.args[0])

    @property
    def lagrangian(self):
        """
        Terminant part of L
        :return: Expr
        """
        return span_dict({k: v for k, v in self.lambdas.items() if k not in [i.args[1] for i in self.objectives]})

    @property
    def kwargs(self):
        """
        Kwargs, to re-init
        :return:
        """
        return {
            'name': self.name,
            'objectives': self.objectives,
            'inequations': self.inequations,
            'equations': self.equations,
            'functions': self.functions,
            'params': self.params,
            'dim_dict': self.dim_dict
        }

    @property
    def args(self):
        """
        Args, to re-init
        :return:
        """
        return self.objectives, self.inequations, self.equations, self.functions, self.params, self.dim_dict

    # ECOMOD CORE SOFT
    def euler_equations(self):
        """
        Euler Equations:
        d/dt L_{x'} = L_{x}
        :return: List[Eq]
        """
        ret = []
        for x in self.phases:
            ret.append(euler_mask(self.Lagrangian, x, self.time))
        return ret

    def transversality_conditions(self):
        """
        Transversality conditions:
        L_{x'}(t_i) = (-1)^i l_{x(t_i)}
        :return: List[Eq]
        """
        ret = []
        for x in self.phases:
            # for all t_0, t_1
            ret.extend(transversality_mask(self.Lagrangian, x, self.time, self.lagrangian, *self.time_horizon))

        return ret

    def control_optimality(self):
        """
        Control optimality conditions, using smoothness of L function wrt control variables
        :return: List[Eq]
        """
        from sympy import Eq
        return [Eq(self.Lagrangian.diff(c), 0) for c in self.controls]

    def KKT(self):
        """
        Dual feasibility and Complementary slackness conditions
        :return: List[Expr]
        """
        _duals = {k: v for k, v in self.duals.items() if k not in [eq2func(e) for e in self.transitions]}
        _lambdas = {k: v for k, v in self.lambdas.items() if k not in [o.rhs for o in self.objectives]}
        return KKT_mask(_duals) + KKT_mask(_lambdas)

    def diff_degree(self, deg=None):
        """
        Returns all expr with Derivative degree == deg.
        :param deg: int. Derivative degree of returning equations
        :return: Dict[Expr->deg] if None else Dict[Expr->`deg`]
        """
        ret = {}
        for eq in self.equations:
            ret[eq] = deriv_degree(eq)
        # casting values to int
        ret = {k: int(v.__str__()) for k, v in ret.items()}
        if deg is not None:
            ret = [eq for eq, d in ret.items() if d == deg]
            return {k: deg for k in ret}
        return ret

    @classmethod
    @log(comment='\nParsing new agent file.')
    def read_from_tex(cls, f):
        """
        Contructor from .tex files. See Knowledge base to get guidelines.
        :param f: filename or fd
        :return: Agent
        """
        from pathlib import Path
        name = Path(f).stem
        header, raw_model = read_model_from_tex(f)
        model = ecomodify_(raw_model)
        return cls(name, *model)

    @log(comment='Agent ready for economic processing')
    def process(self, skip_validation=False):
        """
        Process model to be used in core methods.
        :param skip_validation: bool -- If True validation process will be skipped
        :return: Union[None, AnyError]
        """
        if not skip_validation:
            self.validate(*self.args)

        self.__generate_duals()
        self.processed = True

    def compress(self, to_tex=False, headers=True):
        """

        :param to_tex: bool. If True -- .tex file will be generated
        :param headers: If True -- .tex file will contain headers
        :return: Union[None, RenderedJinjaTemplate]
        """
        ret = {
            "PHASES": latexify(self.phases, to_str=True),  # because we render in one line
            "CONTROLS": latexify(self.controls, to_str=True),  # same
            "INFOS": latexify(self.external, to_str=True),  # same
            "EULERS": latexify(self.euler_equations()),
            "OPTIMAS": latexify(self.control_optimality()),
            "TRANSVERS": latexify(self.transversality_conditions()),
            "KKT": latexify(self.KKT())
        }
        if not to_tex:
            return ret
        else:
            engine = AgentTemplateEngine()
            if headers:
                return engine.render(ret)
            return engine.render(ret).split(r'\begin{document}')[1].split(r'\end{document}')[0]

    @log(comment='Dumping agent file')
    def dump(self, destination=None):
        """
        Create .tex and PDF files with Agent optimal conditions.
        :param destination: filepath or fd.
        :return: PDF, .tex files saved in `destination`
        """
        if not destination:
            destination = '.'
        engine = AgentTemplateEngine()
        engine.render(self.compress())
        tex_directorypath = Path(destination) / self.name
        tex_filepath = (tex_directorypath / self.name).with_suffix('.tex')
        # Due to bug with .absolute() method in Pathlib
        tex_directorypath = Path(str(tex_directorypath.cwd()) + str(tex_directorypath))
        tex_filepath = Path(str(tex_filepath.cwd()) + str(tex_filepath))
        engine.dump(tex_filepath)
        exec_tex(tex_filepath, tex_directorypath)

