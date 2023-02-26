import yaml
from core.errors.RWErrors import NonLaTeXableError, NonYAMLableError
from core.utils import unpack, trim


def read_tex(f):
    """
    Read tex file and split to `yamlable` and header
    :param f: filename or fd
    :return: yaml parsable object + header
    """
    with open(f, 'r', encoding='utf-8') as stream:
        content = stream.read()
        splitted = content.split(r'\begin{document}')
        if len(splitted) == 1:
            raise NonLaTeXableError(file=f)
        header = splitted[0]
        content = splitted[1].split(r'\end{document}')[0]
    return content, header


def parse_yaml(stream: str):
    """
    Simple YAML decoder, to distinguish errors.
    :param stream: string value after reducing tex-header
    :return: dict: decoded yaml
    """
    try:
        decoded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise NonYAMLableError(err=exc)

    return decoded


def read_model_from_tex(f):
    """
    Full blackbox model parsing from .tex file. Used as subfunction in Agent.read_from_tex
    :param f: filename or fd
    :return: .tex header and KV-storage (json-like) with raw model
    """
    # step 1: extract latex headers
    content, header = read_tex(f)
    # step 2: extract yaml data
    raw_model = parse_yaml(content)
    raw_model = unpack(raw_model)
    raw_model_ = {trim(k): v for k, v in raw_model.items()}
    return header, raw_model_


def main():
    # check for test model 1
    f = './inputs/agent.tex'
    h, r = read_model_from_tex(f)
    with open('text.txt', 'w') as f:
        f.write(h)
        f.write(r.__str__())


if __name__ == "__main__":
    main()
