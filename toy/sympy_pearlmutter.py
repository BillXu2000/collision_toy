import sympy

sympy.Mul

def getvars(f, constant=set()):
    vars = set()
    for i in sympy.preorder_traversal(f):
        if i.func in constant: continue
        if i.func == sympy.Symbol:
            if '[' not in i.name: continue
            if 'data' in i.name: continue
            vars.add(i)
    vars = list(vars)
    vars.sort(key=lambda x: x.name)
    return vars

def get_partial(e, is_constant = lambda x: False):
    hack = sympy.Symbol('(Hack)')
    vars = set()
    for i in sympy.preorder_traversal(e):
        if isinstance(i, sympy.Symbol) and not is_constant(i):
            vars.add(i)
    for y in vars:
        e = e.replace(y, sympy.Function(f'f(hack){y.name}')(hack))
    pe = e.diff(hack)
    for y in vars:
        pe = pe.replace(sympy.Function(f'f(hack){y.name}')(hack).diff(hack), sympy.Symbol(r'\partial %s' % y.name))
        pe = pe.replace(sympy.Function(f'f(hack){y.name}')(hack), y)
    return pe

class Cross(sympy.Function):
    @classmethod
    def eval(cls, x, y):
        if x == 0 or y == 0:
            return 0

    def _eval_derivative(self, s):
        args = list(self.args)
        return Cross(args[0].diff(s), args[1]) + Cross(args[0], args[1].diff(s))
    
    def _latex(self, printer):
        _args = [printer._print(i) for i in self.args]
        return r'%s \times %s' % tuple(_args)

class Dot(sympy.Function):
    def _latex(self, printer):
        _args = [printer._print(i) for i in self.args]
        return r'%s \cdot %s' % tuple(_args)

class Norm(sympy.Function):
    def _eval_derivative(self, s):
        v = self.args[0]
        return Dot(v, v.diff(s)) / Norm(v)
    
    def _latex(self, printer, exp=0):
        _v = printer._print(self.args[0])
        ans = r'\| %s \|' % _v
        if exp != 0:
            ans += r'^{%s}' % str(exp)
        return ans

def pearlmutter_sympy(f, constant=set()):
    vars = getvars(f, constant)
    df = {}
    ddf = {}
    for i, x in enumerate(vars):
        dfdx = sympy.diff(f, x)
        df[x] = sympy.nsimplify(dfdx)
        ddf[x] = get_partial(df[x])
        # ddf[x] = sympy.nsimplify(ddfdx)
    # return sympy.Matrix(m).transpose()
    return df, ddf

def norm2(x):
    return sum([i**2 for i in x])**0.5
    
def normsqr(x):
    return sum([i**2 for i in x])

def vector_sympy(name, n):
    return sympy.matrices.Matrix([sympy.Symbol(name % i) for i in range(n)])

def sympy2str(f):
    # f = sympy.nsimplify(f)
    # f = f.replace(normsqr(x_global), sympy.Symbol('x.norm_sqr()'))
    s = str(f)
    s = s.replace('sqrt', 'ti.sqrt')
    return s

class SympyTreeCounter:
    cache = {}
    @classmethod
    def get_size(cls, e):
        if e in cls.cache:
            return cls.cache[e]
        size = sum([cls.get_size(i) for i in e.args]) + 1
        cls.cache[e] = size
        return size

class subst:
    def __init__(self):
        self.visit = {}
        self.sub = []
        self.exp2sub = {}
        self.processed = False
    
    def record(self, exp):
        self.processed = False
        if len(exp.args) == 0:
            return
        self.visit[exp] = self.visit.get(exp, 0) + 1
        if self.visit[exp] >= 2:
            if self.visit[exp] == 2:
                self.sub.append(exp)
            return
        for i in exp.args:
            self.record(i)
    
    def process(self):
        self.sub.sort(key=lambda e: SympyTreeCounter.get_size(e))
        for i, a in enumerate(self.sub):
            self.exp2sub[a] = i
            print(i, a, SympyTreeCounter.get_size(a))
        self.processed = True
    
    def translate(self, exp, root=False):
        if not self.processed:
            self.process()
        if len(exp.args) == 0:
            return exp
        if exp in self.exp2sub and not root:
            return sympy.Symbol(f"tmp{self.exp2sub[exp]}")
        return exp.func(*[self.translate(i) for i in exp.args])

def target2ti(target):
    ti_template = '''
import taichi as ti
@ti.func
def f(data, i, x):
#replacef

@ti.func
def df(data, i, x):
#replacedf

@ti.func
def ddf(data, i, x, dx):
#replaceddf
'''
    # print(df, ddf)
    target = sympy.simplify(target)
    target = sympy.nsimplify(target)
    df, ddf = pearlmutter_sympy(target)
    vars = getvars(target)
    indent = ' ' * 4
    output = ti_template
    output = output.replace('#replacef', f'{indent}return {target}')
    # print(output)

    sub_df = subst()
    for x in vars:
        sub_df.record(df[x])
    sub_df.process()
    str_df = f'{indent}dfdx = x\n'
    for i, sub in enumerate(sub_df.sub):
        str_df += f'{indent}tmp{i} = {sympy2str(sub_df.translate(sub, root=True))}\n'
        # print(sub_df.translate(sub, root=True))
    for x in vars:
        dfdx = df[x]
        str_df += f'{indent}dfd{x} = {sympy2str(sub_df.translate(dfdx))}\n'
    str_df += f'{indent}return dfdx\n'
    output = output.replace('#replacedf', str_df)




    sub_ddf = subst()
    for x in vars:
        sub_ddf.record(ddf[x])
    sub_ddf.process()
    str_ddf = f'{indent}ddfdx = x\n'
    for i, sub in enumerate(sub_ddf.sub):
        str_ddf += f'{indent}tmp{i} = {sympy2str(sub_ddf.translate(sub, root=True))}\n'
    for x in vars:
        ddfdx = ddf[x]
        str_ddf += f'{indent}ddfd{x} = {sympy2str(sub_ddf.translate(ddfdx))}\n'
        # str_ddf += f'{indent}#ddfd{x} = {sympy.latex(ddfdx)}\n'
    str_ddf += f'{indent}return ddfdx\n'
    output = output.replace('#replaceddf', str_ddf)
    return output

def bending_force():
    x = vector_sympy('x[%d]', 3)
    xn = x / sum([x[i]**2 for i in range(3)])**0.5
    ti = vector_sympy('data[%d]', 3)
    kb = 2 * xn.cross(ti) / (1 + x.dot(ti))
    target = kb.dot(kb) * (1 + xn.dot(ti))**2 / (1 + xn.dot(ti))**2
    target = sympy.nsimplify(target)
    target = sympy.simplify(target)
    # return target
    print(target)
    ans = target2ti(target)
    print(ans)

def default_target():
    # global x_global
    x = vector_sympy('x[%d]', 3)
    # x_global = x
    # x = vector_sympy('x[0, %d]', 3)
    y = vector_sympy('x[1, %d]', 3)
    z = vector_sympy('x[2, %d]', 3)
    # x = vector_sympy('x_%d', 3)
    # y = vector_sympy('y_%d', 3)
    # z = vector_sympy('z_%d', 3)
    # target = norm2(x)
    # target = x.dot(y.cross(z)) / norm2(y) / norm2(z)
    # target = x.dot(y.cross(z))
    target = sympy.Symbol('data.k') / 2 * (norm2(x) - sympy.Symbol('data.l_0[i]'))**2
    target = sympy.nsimplify(target)
    target = sympy.simplify(target)
    print(target)
    print(get_partial(target, is_constant=lambda x: 'data' in x.name))

    # ans = target2ti(target)
    # print(ans)

if __name__ == '__main__':
    # bending_force()
    default_target()
    