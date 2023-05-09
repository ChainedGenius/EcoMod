============
Integrations
============

-----
Latex
-----
https://ecomod.readthedocs.io/en/latest/index.html
Warning: working in test mode

Conclusion module

.. code:: python
    class TexTemplateEngine(object):
    """
    Template engine to produce TeX and #PDF# file for Agent processed output.

    Methods:

        1. get_template_variables
            Parameters which are included in Template produced from `self.template_name`
        2. render
            Template rendering process.
        3. dump
            Write rendered template into file
    """
    template_name = 'basic.tex'

    def __init__(self):
        self.latex_jinja_env = Environment(
            block_start_string='\BLOCK{',
            block_end_string='}',
            variable_start_string='\VAR{',
            variable_end_string='}',
            comment_start_string='\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            autoescape=False,
            loader=FileSystemLoader('templates')
        )
        self.template = self.latex_jinja_env.get_template(self.template_name)
        self.rendered = ""

    @property
    def get_template_variables(self):
        """
        Get reserved variables from template, class_attr template_name
        :return: List[]
        """
        template_source = self.latex_jinja_env.loader.get_source(self.latex_jinja_env, self.template_name)[0]
        parsed_content = self.latex_jinja_env.parse(template_source)
        return meta.find_undeclared_variables(parsed_content)

    def render(self, data):
        """
        Render .tex template with given KV-storage `data`
        :param data: Dict[K -> Render_value]
        :return: RenderedTemplate
        """
        # data is kv-storage which contains data for render
        # kindly check if there are redundant keys
        data = {k: v for k, v in data.items() if k in self.get_template_variables}
        self.rendered = self.template.render(**data)
        return self.rendered

    def dump(self, filename):
        """
        Save .tex
        :param filename: filename or fd
        :return: None
        """
        if not self.rendered:
            raise NotRendered(model=filename.stem)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        with open(filename, 'w') as f:
            f.write(self.rendered)

.. code:: python
    class AgentTemplateEngine(TexTemplateEngine):
        """
        Basic template for one-Agent model.
        """
        template_name = 'LAgent.tex'

.. code:: python
    class ModelTemplateEngine(TexTemplateEngine):
        """
        Template for multi-Agent model.
        """
        template_name = 'Model.tex'



Model entry

----
Geko
----
under development

---------
Hypernetx
---------

Model visualization

.. code:: python
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

Model entry under development
