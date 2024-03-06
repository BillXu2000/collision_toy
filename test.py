import base64
import numpy as np
import io
import toy

# t = np.arange(25, dtype=np.float64)
t = np.array([['233', 2] + [0]*10000, [3, 4] + [0]*10000])
b64 = toy.export.np2b64(t)
q = toy.export.b642np(b64)
# s = base64.b64encode(t)
# r = base64.decodebytes(s)
# q = np.frombuffer(r, dtype=np.float64)

print(q)
print(t)
print(b64)
print('instance', isinstance(q, np.ndarray))
# print(np.allclose(q, t))
