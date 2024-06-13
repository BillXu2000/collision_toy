from .gui import run
from . import force, solver
import numpy as np
from . import sympy_pearlmutter as spm
import os, importlib, hashlib

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
                ans[i] = k.to_numpy() if i != 'n' else k[None]
        return ans

def string2module(s):
    hash = hashlib.sha256(s.encode()).hexdigest()
    folder = os.path.expanduser('~/.cache/bx2k/module_hack/')
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, f'{hash}.py')
    with open(fn, 'w') as fi:
        fi.write(s)
    spec = importlib.util.spec_from_file_location(hash, fn)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo