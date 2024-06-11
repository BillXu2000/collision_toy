import sympy, ast

class Var:
    num = 0
    vars = {}
    pm = {}

    def __init__(self, func, args, expr=None):
        self.func = func
        self.args = args
        self.num = Var.num
        Var.num += 1
    
    def from_sympy(expr):
        if not isinstance(expr, sympy.Basic):
            expr = sympy.sympify(expr)
        if expr.func == sympy.Symbol:
            if expr.name in Var.vars:
                return Var.vars[expr.name]
        ans = Var(expr.func, [Var.from_sympy(i) for i in expr.args])
        if len(expr.args) == 0:
            ans.expr = expr
        if expr.func == sympy.Symbol:
            Var.vars[expr.name] = ans
        return ans
    
    def __add__(self, other):
        if not isinstance(other, Var):
            other = Var.from_sympy(other)
        return Var(sympy.Add, [self, other])

    def __mul__(self, other):
        if not isinstance(other, Var):
            other = Var.from_sympy(other)
        return Var(sympy.Mul, [self, other])
    
    def __pow__(self, other):
        if not isinstance(other, Var):
            other = Var.from_sympy(other)
        return Var(sympy.Pow, [self, other])

    def __neg__(self):
        return -1 * self
    
    def __sub__(self, other):
        return self + (-1 * other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __hash__(self):
        return self.num
    
    def to_sympy(self):
        # if len(self.args) == 0:
        if hasattr(self, 'expr'):
            return self.expr
        return self.func(*[i.to_sympy() for i in self.args])
    
    def __str__(self):
        if self.func == sympy.Add:
            if len(self.args) == 0:
                return '0'
            return ' + '.join([f'({str(i)})' for i in self.args])
        if self.func == sympy.Mul:
            return ' * '.join([f'({str(i)})' for i in self.args])
        if self.func == sympy.Pow:
            return f'({str(self.args[0])})^({str(self.args[1])})'
        if self.func == sympy.log: 
            return f'log({str(self.args[0])})'
        return str(self.expr)
        

def back_propagation(v):
    pre = {}
    def get_pre(v):
        for i in v.args:
            if i not in pre:
                pre[i] = 0
            pre[i] += 1
            get_pre(i)
    get_pre(v)
    # print(pre)
    
    ans = dict([(i, []) for i in pre])
    ans[v] = Var.from_sympy(1)
    # print(ans)
    def get_diff(v):
        if v.func == sympy.Pow:
            ans[v.args[0]].append(ans[v] * v.args[1] * v.args[0]**(v.args[1] - 1))
            ans[v.args[1]].append(ans[v] * v * Var(sympy.log, [v.args[0]]))
        if v.func == sympy.log:
            ans[v.args[0]].append(ans[v] * v.args[0]**-1)
        for i, j in enumerate(v.args):
            if v.func == sympy.Add:
                # print(j)
                ans[j].append(ans[v])
            if v.func == sympy.Mul:
                tmps = [ans[v]]
                for ik, k in enumerate(v.args):
                    if ik != i:
                        tmps.append(k)
                ans[j].append(Var(sympy.Mul, tmps))
            if len(ans[j]) == pre[j]:
                ans[j] = Var(sympy.Add, ans[j])
                get_diff(j)
    get_diff(v)
    return ans

def pearlmutter(v):
    if v in Var.pm:
        return Var.pm[v]
    if v.func == sympy.Symbol:
        return Var.from_sympy(sympy.Symbol('(\partial {%s})' % v.expr.name))
    if v.func == sympy.Pow:
        Var.pm[v] = v.args[1] * v.args[0]**(v.args[1] - 1) * pearlmutter(v.args[0]) + v * Var(sympy.log, [v.args[0]]) * pearlmutter(v.args[1])
        return Var.pm[v]
    if v.func == sympy.log:
        Var.pm[v] = v.args[0]**-1 * pearlmutter(v.args[0])
    ans = []
    for i, j in enumerate(v.args):
        if v.func == sympy.Add:
            ans.append(pearlmutter(j))
        if v.func == sympy.Mul:
            tmps = []
            for ik, k in enumerate(v.args):
                if ik == i:
                    tmps.append(pearlmutter(k))
                else:
                    tmps.append(k)
            ans.append(Var(sympy.Mul, tmps))
    return Var(sympy.Add, ans)

def norm2(x):
    return sum([i**2 for i in x])**0.5

# x = sympy.Matrix([sympy.Symbol('x_%d' % i) for i in range(3)])
# n = Var.from_sympy(norm2(x))
# print(n.to_sympy())
def vector_sympy(name, n):
    return sympy.Matrix([sympy.symbols(f'{name}_{i}') for i in range(n)])


x = vector_sympy('x', 3)
y = vector_sympy('y', 3)
z = vector_sympy('z', 3)
dist = x.dot(y.cross(z)) / norm2(y) / norm2(z)
dist = Var.from_sympy(dist)
print(dist)

bp_ans = back_propagation(dist)

x0 = Var.from_sympy('x_0')
# print(x0.to_sympy(), bp_ans[x0].to_sympy())
# print(x0, bp_ans[x0].to_sympy(), pearlmutter(bp_ans[x0]).to_sympy())
pmx0 = pearlmutter(bp_ans[x0])
print(pmx0)
print()
print(pmx0.to_sympy())
print(sympy.nsimplify(pmx0.to_sympy()))

# def isnumber(v):
#     if hasattr(v, 'expr') and isinstance(v.expr, sympy.Number):
#         return True
#     return False

# def simplify(v):
#     if hasattr(v, 'expr'):
#         return v
#     args = []
#     have_var = False
#     for i in v.args:
#         args.append(simplify(i))
#         if not isnumber(i):
#             have_var = True
#     if not have_var:
#         return Var.from_sympy(v.to_sympy())
#     if v.func == sympy.Mul:
#         for i in args:
#             if i == 0:
#                 return Var.from_sympy(0)
#     return Var(v.func, args)
# print(str(pmx0.to_sympy()).replace('**', '^'))
# print(str(pmx0))
# print(str(simplify(pmx0)))
# print(sympy.nsimplify(pmx0.to_sympy()))


# for i in bp_ans:
    
#     print(i.to_sympy(), bp_ans[i].to_sympy(), pearlmutter(bp_ans[i]).to_sympy())
#     # dfs(ans[i])

