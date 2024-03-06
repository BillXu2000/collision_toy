from .gui import run
from . import force
import numpy as np

class State:
    def __init__(self):
        self.names = ['x', 'v', 'mass', 'n']

    def loads(self, d):
        for i in d:
            k = getattr(self, i, None)
            if k is not None:
                k.from_numpy(np.array(d[i]))

    def dumps(self):
        ans = {}
        for i in self.names:
            k = getattr(self, i, None)
            if k is not None:
                ans[i] = k.to_numpy()
        return ans
