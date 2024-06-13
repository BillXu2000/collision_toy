import taichi as ti, numpy as np, hashlib
import importlib
import toy
ti.init()

test_code = '''
import taichi as ti
@ti.kernel
def test():
    for i in range(1):
        a = ti.Matrix([[1, 2], [3, 4]])
        b = a
        b[1, :] *= 0
        c = b[1, :]
        print(c[0])
'''

code2 = '''
def ra():
    return 0
'''

code_numpy = '''
def ra():
    return np.array([233])
'''

# def string2module(s):
#     hash = hashlib.sha256(s.encode()).hexdigest()
#     fn = f'./.cache/{hash}.py'
#     with open(fn, 'w') as fi:
#         fi.write(s)
#     spec = importlib.util.spec_from_file_location(hash, fn)
#     foo = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(foo)
#     return foo


# foo = toy.string2module(test_code)
# print(foo)


# exec(test_code)

# print(dir(test))
# test()
# foo.test()

@ti.kernel
def run():
    for i in range(1):
        a = ti.Matrix([[1, 2], [3, 4]])
        b = ti.Vector([1, 2])
        print(a[1, :] - b)

run()

