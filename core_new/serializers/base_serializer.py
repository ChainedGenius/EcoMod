import pickle


class BaseSerializer:
    def __init__(self, file):
        self.file = file

    @classmethod
    def serialize(cls, f):
        from pathlib import Path
        name = Path(f).stem
        with open(f, 'r') as stream:
            return cls._serialize(stream), name

    @classmethod
    def _serialize(cls, stream):
        return pickle.loads(stream)
