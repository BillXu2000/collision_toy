import taichi as ti

@ti.func
def poly(c, x):
    ans = 0.0
    for i in ti.static(range(c.n)):
        ans += c[i] * x**i
    return ans

@ti.func
def d_poly(c, x):
    ans = 0.0
    for i in ti.static(range(1, c.n)):
        ans += c[i] * x**(i - 1) * i
    return ans

@ti.func
def solve_quadratic(c):
    n = 0
    root = ti.Vector([0.0, 0.0])
    d = c[1]**2 - 4 * c[0] * c[2]
    if d > 0:
        q = -(c[1] + ti.math.sign(c[1]) * d**0.5) / 2
        if c[2] != 0:
            root[n] = q / c[2]
            n += 1
        if q != 0:
            root[n] = c[0] / q
            n += 1
        if n == 2 and root[0] > root[1]:
            root[0], root[1] = root[1], root[0]
    return n, root

@ti.func
def newton_1root(c, l, r):
    f_l = poly(c, l)
    eps = 1e-6
    if f_l > 0:
        f_l = 1
    else:
        f_l = -1
    last_root = l
    root = (l + r) / 2
    iter = 0
    while iter < 10 or ti.abs(root - last_root) > eps:
        iter += 1
        err = poly(c, root)
        gradient = d_poly(c, root)
        if err * f_l > 0:
            l = root
        else:
            r = root
        last_root = root
        if gradient == 0:
            root = (l + r) / 2
        else:
            root -= err / gradient
            if root < l or root > r:
                root = (l + r) / 2
    return root

@ti.func
def solve_cubic_ranged(c, l, r):
    m, y = solve_quadratic([c[1], 2 * c[2], 3 * c[3]])
    x = ti.Vector([0.0] * 4)
    x[0] = l
    n = 1
    for i in range(m):
        if l < y[i] < r:
            x[n] = y[i]
            n += 1
    x[n] = r
    n += 1
    val = ti.Vector([0.0] * 4)
    for i in range(n):
        val[i] = poly(c, x[i])
    n_root = 0
    root = ti.Vector([0.0] * 3)
    for i in range(n - 1):
        if val[i] * val[i + 1] < 0:
            root[n_root] = newton_1root(c, x[i], x[i + 1])
            n_root += 1
    return n_root, root