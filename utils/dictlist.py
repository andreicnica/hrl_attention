"""
    Copyright (c) https://github.com/lcswillems/torch-rl
"""
from typing import Any


class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList(dict({"a": [[1, 2], [3, 4]], "b": [[5], [6]]}))
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self) -> int:
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index: int) -> super:
        return DictList(dict({key: value[index] for key, value in self.items()}))

    def __setitem__(self, index: int, d: Any):
        dict.__setitem__(self, index, d)

    def __setstate__(self, d):
        for key, value in d.items():
            dict.__setitem__(self, key,  value)

    def __getstate__(self):
        return dict({key: value for key, value in self.items()})
