import sympy

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

def pearlmutter_sympy(f, constant=set()):
    vars = getvars(f, constant)
    df = {}
    ddf = {}
    for i, x in enumerate(vars):
        dfdx = sympy.diff(f, x)
        df[x] = dfdx
        hack = sympy.Symbol('(Hack)')
        for y in vars:
            dfdx = dfdx.replace(y, sympy.Function(f'f{y.name}')(hack))
        # print(df)
        ddfdx = dfdx.diff(hack)
        for y in vars:
            ddfdx = ddfdx.replace(sympy.Function(f'f{y.name}')(hack).diff(hack), sympy.Symbol('d%s' % y.name))
            ddfdx = ddfdx.replace(sympy.Function(f'f{y.name}')(hack), y)
        ddf[x] = ddfdx
    # return sympy.Matrix(m).transpose()
    return df, ddf

def norm2(x):
    return sum([i**2 for i in x])**0.5
    
def normsqr(x):
    return sum([i**2 for i in x])

def vector_sympy(name, n):
    return sympy.Matrix([sympy.Symbol(name % i) for i in range(n)])

def sympy2str(f):
    f = sympy.nsimplify(f)
    # f = f.replace(normsqr(x_global), sympy.Symbol('x.norm_sqr()'))
    s = str(f)
    s = s.replace('sqrt', 'ti.sqrt')
    return s


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
    df, ddf = pearlmutter_sympy(target)
    vars = getvars(target)
    indent = ' ' * 4
    output = ti_template
    output = output.replace('#replacef', f'{indent}return {target}')
    # print(output)
    str_df = f'{indent}dfdx = x\n'
    for x in vars:
        dfdx = df[x]
        str_df += f'{indent}dfd{x} = {sympy2str(dfdx)}\n'
    str_df += f'{indent}return dfdx\n'
    output = output.replace('#replacedf', str_df)
    str_ddf = f'{indent}ddfdx = x\n'
    for x in vars:
        ddfdx = ddf[x]
        str_ddf += f'{indent}ddfd{x} = {sympy2str(ddfdx)}\n'
        # str_ddf += f'{indent}#ddfd{x} = {sympy.latex(ddfdx)}\n'
    str_ddf += f'{indent}return ddfdx\n'
    output = output.replace('#replaceddf', str_ddf)
    return output

if __name__ == '__main__':
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

    ans = target2ti(target)
    print(ans)