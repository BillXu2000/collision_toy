import types, taichi as ti

ti.init()

def import_code(code, name):
    # create blank module
    module = types.ModuleType(name)
    # populate the module with code
    exec(code, module.__dict__)
    return module

code = """
import taichi as ti
@ti.kernel
def testFunc():
    print('spam!')
"""

m = import_code(code, 'test')
m.testFunc()