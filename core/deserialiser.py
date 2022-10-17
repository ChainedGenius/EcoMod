import yaml
from errors.RWErrors import NonLaTeXableError, NonYAMLableError


def read_tex(f):
    """
    :param f: filename or fd
    :return: yaml parsable object + header
    """
    with open(f, 'r') as stream:
        content = stream.read()
        splitted = content.split(r'\begin{document}')
        if len(splitted) == 1:
            raise NonLaTeXableError({"file": f})
        header = splitted[0]
        content = splitted[1].split(r'\end{document}')[0]
    return content, header


def parse_yaml(stream: str):
    """
    :param stream: string value after reducing tex-header
    :return: dict: decoded yaml
    """
    try:
        decoded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise NonYAMLableError({"err": exc})

    return decoded


def read_model_from_tex(f):
    # step 1: extract latex headers
    content, header = read_tex(f)
    # step 2: extract yaml data
    raw_model = parse_yaml(content)
    return header, raw_model


def main():
    # check for test model 1
    f = './inputs/agent.tex'
    h, r = read_model_from_tex(f)
    with open('text.txt', 'w') as f:
        f.write(h)
        f.write(r.__str__())


if __name__ == "__main__":
    main()
