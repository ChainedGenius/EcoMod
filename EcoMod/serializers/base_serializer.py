import pickle

from EcoMod.utils import global_path, get_root_dir

root_dir = get_root_dir()

class BaseSerializer:
    def __init__(self, file):
        self.file = file

    @classmethod
    def serialize(cls, f):
        from pathlib import Path
        name = Path(f).stem
        f_ = root_dir + f

        with open(f_, 'r') as stream:
            return cls._serialize(stream), name

    @classmethod
    def _serialize(cls, stream):
        return pickle.loads(stream)
