class BaseSerializer:
    def __init__(self, file):
        self.file = file

    @classmethod
    def serialize(cls, f):
        with open(f, 'r') as stream:
            return cls._serialize(stream)

    @classmethod
    def _serialize(cls, f):
        pass
