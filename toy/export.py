import time, json, io, base64, numpy as np, zlib

def np2b64(t):
    f = io.BytesIO()
    np.save(f, t)
    return base64.b64encode(zlib.compress(f.getbuffer())).decode()

def b642np(s):
    return np.load(io.BytesIO(zlib.decompress(base64.b64decode(s))))

class __Exporter:
    def __init__(self):
        self.fn = './log/' + '%.3f' % time.time() + '.log'
        self.i_f = -1

    def set_i_f(self, i_f):
        self.i_f = i_f

    def export(self, d):
        assert isinstance(d, dict)
        for i in d:
            if isinstance(d[i], np.ndarray):
                d[i] = np2b64(d[i])
        d['i_f'] = self.i_f
        with open(self.fn, 'a') as fi:
            fi.write(json.dumps(d) + '\n')

exporter = __Exporter()