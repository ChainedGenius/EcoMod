import pickle

from EcoMod.utils import global_path, get_root_dir

root_dir = get_root_dir()


class BaseSerializer:
    def __init__(self, file):
        self.file = file

    def serialize(self):
        from pathlib import Path
        name = Path(self.file).stem
        f_ = root_dir + self.file

        with open(f_, 'r') as stream:
            return self._serialize(stream), name

    def _serialize(self, stream):
        return pickle.loads(stream)
