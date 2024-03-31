import yaml
from sympy.parsing.latex import parse_latex

from core_new.serializers.base_serializer import BaseSerializer
from errors.RWErrors import NonYAMLableError
from core_new.utils import trim, unpack


class LaTeXSerializer(BaseSerializer):
    @classmethod
    def _serialize(cls, stream):
        content, header = cls.__split_header_content(stream)
        decoded = cls.__yamlable_content(content)
        return {parse_latex(trim(k)): v for k, v in unpack(decoded).items()}

    @staticmethod
    def __split_header_content(stream):
        content = stream.read()
        splitted = content.split(r'\begin{document}')
        if len(splitted) == 1:
            raise TypeError('No begin and end doc')  # NonLaTeXableError()
        header = splitted[0]
        content = splitted[1].split(r'\end{document}')[0]
        return content, header

    @staticmethod
    def __yamlable_content(content):
        try:
            decoded = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise NonYAMLableError(err=exc)

        return decoded
