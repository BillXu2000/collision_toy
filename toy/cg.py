import taichi as ti

@ti.kernel
def ax_by(z: ti.template(), a: ti.f32, x: ti.template(), b: ti.f32, y: ti.template()):
    for i in range(newton.n[None]):
        z[i] = a * x[i] + b * y[i]

@ti.kernel
def ax_by_s(z: ti.template(), a: ti.f32, x: ti.template(), b: ti.f32, y: ti.template(), status: ti.template()):
    for i in z:
        if status[i] & constant.FIXED: continue
        z[i] = a * x[i] + b * y[i]

@ti.kernel
def vec_mul(z: ti.template(), x: ti.template(), y: ti.template()):
    for i in z:
        z[i] = x[i] * y[i]

@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> ti.f32:
    ans = 0.0
    for i in range(newton.n[None]):
        ans += a[i].dot(b[i])
    return ans

# @ti.kernel
# def clean_r(r: ti.template(), status: ti.template()):
#     for i in r:
#         if status[i] & constant.FIXED: r[i] = [0, 0, 0]

# def cg(A, x, b, r, p):
#     r.fill(0)
#     ax_by(r, 1, b, -1, A(x))
#     p.copy_from(r)
#     r_2 = dot(r, r)
#     r_0 = r_2
#     n_iter = 50
#     eps = 1e-9
#     if r_0 < 1e-20: return
#     for iter in range(n_iter):
#         A_p = A(p)
#         dot_ans = dot(p, A_p)
#         alpha = r_2 / dot_ans
#         ax_by(x, 1, x, alpha, p)
#         ax_by(r, 1, r, -alpha, A_p)
#         r_2_new = dot(r, r)
#         # if r_2_new / r_0 < eps: break
#         if r_2_new < eps: break
#         beta = r_2_new / r_2
#         ax_by(p, 1, r, beta, p)
#         r_2 = r_2_new

# def newton(f, x0, x, b, r, p):
#     n_iter = 5
#     for iter in range(n_iter):
#         b = f(x0, b)
#         ax_by(b, -1, b, 0, b)
#         cg()

class newton:
    def __init__(self, gen_field):
        self.gen_field = gen_field
        self.b = gen_field()
        self.r = gen_field()
        self.p = gen_field()
        self.dx = gen_field()
        self.A_p = gen_field()
        self.pos = gen_field()
    
    def newton(self, energy, f, df, x0):
        pos = self.pos
        b = self.b
        pos.copy_from(x0)
        n_iter = 3
        for iter in range(n_iter):
            f(b, pos)
            # print(f'iter = {iter}, b = {b.to_numpy()[:5]}')
            ax_by(b, -1, b, 0, b)
            def A(ans, dx):
                return df(ans, pos, dx)
            dx = self.cg(A, b)
            # print(f'iter = {iter}, dx = {dx.to_numpy()[:5]}')
            e_0 = energy(pos)
            x_1 = b
            alpha = 1.0
            def ene():
                ax_by(x_1, 1, pos, alpha, dx)
                ans = energy(x_1)
                # print(f'alpha = {alpha}, ans = {ans}, e_0 = {e_0}')
                return ans
            while not ene() <= e_0:
                alpha /= 2
            d_e = ene() - e_0
            ax_by(pos, 1, pos, alpha, dx)
            mi = pos.to_numpy()[:3, 1].min()
            # print(f'iter = {iter}, alpha = {alpha}, min = {mi}, e_0 = {e_0}, d_e = {d_e}')
            # if mi <= 1e-3: exit(0)
        return pos
    
    def cg(self, A, b, x0=None):
        x = self.dx
        if x0 is not None:
            x.copy_from(x0)
        else:
            x.fill(0)
        r = self.r
        p = self.p
        A_p = self.A_p
        A(A_p, x)
        ax_by(r, 1, b, -1, A_p)
        p.copy_from(r)
        r_2 = dot(r, r)
        r_0 = r_2
        n_iter = 10
        eps = 1e-10
        if r_0 < 1e-20: return
        for iter in range(n_iter):
            A(A_p, p)
            dot_ans = dot(p, A_p)
            if dot_ans < eps: break
            alpha = r_2 / dot_ans
            ax_by(x, 1, x, alpha, p)
            ax_by(r, 1, r, -alpha, A_p)
            r_2_new = dot(r, r)
            # if r_2_new / r_0 < eps: break
            if r_2_new < eps: break
            beta = r_2_new / r_2
            ax_by(p, 1, r, beta, p)
            r_2 = r_2_new
        # print('cg', iter, r_2)
        A(A_p, x)
        ax_by(r, 1, b, -1, A_p)
        err = dot(r, r)
        # if not err < 1e-7: print(iter, err, r_2)
        return x