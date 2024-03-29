import yaml

from core_new.serializers.base_serializer import BaseSerializer
from errors.RWErrors import NonYAMLableError


class LaTeXSerializer(BaseSerializer):
    @classmethod
    def _serialize(cls, stream):
        content, header = cls.__split_header_content(stream)


    @staticmethod
    def __split_header_content(stream):
        content = stream.read()
        splitted = content.split(r'\begin{document}')
        if len(splitted) == 1:
            raise TypeError('No begin and end doc')#NonLaTeXableError()
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

if __name__ == '__main__':
    a = LaTeXSerializer.serialize(r'C:\Users\BaTe\Documents\GitHub\EcoMod\models\inputs\agent.tex')
    print(a)