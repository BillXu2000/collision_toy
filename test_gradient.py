import sympy
sympy.init_printing()

l0 = sympy.symbols('l_0')
k = sympy.symbols('k')
n = 3

def norm(x):
    ans = 0
    for i in x:
        ans += i**2
    return sympy.sqrt(ans)

def dist(x):
    return norm(x) - l0

def energy(x):
    return k * dist(x)**2 / 2

xs = [sympy.symbols(f'x_{i}') for i in range(n)]
print(xs)
print(dist(xs))
print(energy(xs))
df = 0
for i, x in enumerate(xs):
    df += sympy.symbols(f'dx_{i}') * sympy.diff(energy(xs), x)
df = sympy.simplify(df)
print(df)