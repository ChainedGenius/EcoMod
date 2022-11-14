from os import startfile
from os.path import exists
from pathlib import Path
from shutil import rmtree, move
from subprocess import run
from tempfile import mkdtemp
from jinja2 import Environment, FileSystemLoader, meta


class TexTemplateEngine(object):
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
            loader=FileSystemLoader(Path('./templates').absolute())
        )
        self.template = self.latex_jinja_env.get_template(self.template_name)
        self.rendered = ""

    @property
    def get_template_variables(self):
        template_source = self.latex_jinja_env.loader.get_source(self.latex_jinja_env, self.template_name)[0]
        parsed_content = self.latex_jinja_env.parse(template_source)
        return meta.find_undeclared_variables(parsed_content)

    def render(self, data):
        # data is kv-storage which contains data for render
        # kindly check if there are redundant keys
        data = {k: v for k, v in data.items() if k in self.get_template_variables}
        self.rendered = self.template.render(**data)
        return self.rendered

    def dump(self, filename):
        if not self.rendered:
            # TODO: custom warn
            raise RuntimeError('Render it first')

        with open(filename, 'w') as f:
            f.write(self.rendered)


class AgentTemplateEngine(TexTemplateEngine):
    template_name = 'LAgent.tex'


def exec_tex(tex_filename, destination, open=False):
    SUFFIXES = ['.pdf', '.log', '.aux']

    filename = Path(tex_filename).stem
    package_destination = Path(destination) / filename  # Filepath without suffix
    if not Path(package_destination).exists():
        package_destination.mkdir(parents=True)

    temp_dir = mkdtemp()
    try:
        run(['pdflatex', '-interaction=nonstopmode', tex_filename])
        for suffix in SUFFIXES:
            real_filename = Path(filename).with_suffix(suffix)
            real_file_path = package_destination / real_filename
            move(Path.cwd() / real_filename, real_file_path)
            if suffix == '.pdf':
                pdf_destination = package_destination / real_filename
    finally:
        rmtree(temp_dir)

    if open:
        import platform

        if not exists(pdf_destination):
            raise RuntimeError('PDF output not found')

        if platform.system().lower() == 'darwin':
            run(['open', pdf_destination])
        elif platform.system().lower() == 'windows':
            startfile(pdf_destination)
        elif platform.system().lower() == 'linux':
            run(['xdg-open', pdf_destination])
        else:
            raise RuntimeError('Unknown operating system "{}"'.format(platform.system()))


if __name__ == '__main__':
    # exec_tex('../models/inputs/Pmodel/H.tex', '../models/outputs/Pmodel')
    a = AgentTemplateEngine()
    a.render({'PHASES': '3333', 'CONTROLS': '4444', "INFOS": "6666", "EULERS": [r'\beta +1 = 0', r'\frac{dx(t)}{dt} = 123']})
    a.dump('../models/outputs/Pmodel/H/H.tex')
