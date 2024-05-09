import taichi as ti
import numpy as np, json
from . import cg, export
import math

def loads(args, solver):
    # name = args['class']
    # if name == 'Elasticity': name = 'Elasticity_legacy'
    return globals()[args['class']](args, solver)

@ti.func
def barrier(d: ti.f32, dm: ti.f32) -> ti.f32:
    return -(d - dm)**2 * ti.log(d / dm)

@ti.func
def f_barrier(d: ti.f32, dm: ti.f32) -> ti.f32:
    return -(dm - d) / d * (-dm + 2 * d * ti.log(d / dm) + d) # f = -de/d(d)

@ti.func
def df_barrier(d: ti.f32, dm: ti.f32) -> ti.f32:
    return -((dm / d)**2 + 2 * dm / d - 2 * ti.log(d / dm) - 3)

@ti.func
def sign(d: ti.f32) -> ti.f32:
    ans = 1
    if d < 0: ans = -1
    return ans

@ti.func
def solve2d(coe: ti.template(), x: ti.template(), n: ti.template()):
    a = coe[2]
    b = coe[1]
    c = coe[0]
    delta = b**2 - 4 * a * c
    if delta < 0: 
        n = 0
    else:
        n = 2
        if delta == 0: n = 1
        x[0] = -(2 * c) / (b + sign(b) * ti.sqrt(delta))
        x[1] = -(b + sign(b) * ti.sqrt(delta)) / (2 * a)
        if x[0] > x[1]:
            tmp = x[0]
            x[0] = x[1]
            x[1] = tmp

@ti.func
def check(i, j, k, t, x, dx, dm) -> ti.i32:
    v = (x[i] + dx[i] * t) - (x[j] + dx[j] * t)
    u = (x[k] + dx[k] * t) - (x[j] + dx[j] * t)
    # return -dm < v.dot(u / u.norm_sqr()) < 1 + dm
    return 0 < v.dot(u / u.norm_sqr()) < 1

@ti.func
def collision_test(i, j, k, x, dm, v: ti.template(), d: ti.template()) -> ti.i32:
    flag = 1
    if i == j or i == k: flag = 0
    p = x[i] - x[j]
    q = x[k] - x[j]
    t = p.dot(q / q.norm_sqr())
    if t < -dm or t > 1 + dm: flag = 0
    d_signed = p.cross(q.normalized())
    d = ti.abs(d_signed)
    assert flag == 0 or 0 < d
    if d > dm: flag = 0
    v = ti.Vector([j, i, k], dt=ti.i32)
    if d_signed < 0:
        v[0] = i
        v[1] = j
    return flag
        
@ti.data_oriented
class Collision:
    # def __init__(self, n, k, d_m, links, m_max):
    #     self.n = n
    #     self.k = k
    #     self.d_m = d_m

    #     links = np.array(links)
    #     self.vert = ti.Vector.field(2, dtype=ti.i32, shape=m_max)
    #     self.vert.from_numpy(np.resize(links, (m_max, 2)))
    #     self.m = ti.field(dtype=ti.i32, shape=())
    #     self.m[None] = links.shape[0]

    def __init__(self, args, solver):
        tmp = solver.args.copy()
        tmp.update(args)
        args = tmp

        self.n = solver.n
        self.k = args['k_collision']
        self.d_m = args['d_m']
        m_max = args['m_max']

        links = args['links']
        if isinstance(links, str):
            links = export.b642np(links)
        links = np.array(links)
        self.vert = ti.Vector.field(2, dtype=ti.i32, shape=m_max)
        self.vert.from_numpy(np.resize(links, (m_max, 2)))
        self.m = ti.field(dtype=ti.i32, shape=())
        self.m[None] = links.shape[0]
    
    def dumps(self):
        return {'class': 'Collision', 'links': export.np2b64(self.vert.to_numpy()[:self.m[None]]), 'k_collision': self.k, 'd_m': self.d_m, 'm_max': self.m[None]}
    
    @ti.kernel
    def ccd(self, x: ti.template(), dx: ti.template()) -> ti.f32:
        alpha = 1.0
        verts = ti.static(self.vert)
        for i in range(self.n[None]):
            for j in range(self.m[None]):
                if verts[j][0] == i or verts[j][1] == i: continue
                a0 = (x[verts[j][0]] - x[i]).cross(x[verts[j][1]] - x[i])
                a1 = (dx[verts[j][0]] - dx[i]).cross(x[verts[j][1]] - x[i]) + (x[verts[j][0]] - x[i]).cross(dx[verts[j][1]] - dx[i])
                a2 = (dx[verts[j][0]] - dx[i]).cross(dx[verts[j][1]] - dx[i])
                # if sign(a0) != sign(a0 + a1 + a2): print(i, j)
                if a0 == 0: continue
                # a0 *= 0.99
                xs = ti.Vector([0, 0], dt=ti.f32)
                n_x = 0
                solve2d(ti.Vector([a0, a1, a2]), xs, n_x)
                # if sign(a0) != sign(a0 + a1 + a2): print(i, j, n_x, xs[0], xs[1])
                if n_x == 0: continue
                ans = 2.0
                if 0 < xs[1] < 1:
                    if check(i, verts[j][0], verts[j][1], xs[1], x, dx, self.d_m): ans = xs[1]
                if 0 < xs[0] < 1:
                    if check(i, verts[j][0], verts[j][1], xs[0], x, dx, self.d_m): ans = xs[0]
                if ans == 0: print(i, j, xs[0], xs[1])
                if 0 < ans < 1: ti.atomic_min(alpha, ans)
        return alpha

    @ti.kernel
    def energy(self, x: ti.template(), n: ti.i32) -> ti.f32:
        verts = ti.static(self.vert)
        dm = ti.static(self.d_m)
        ans = .0
        for i in range(self.n[None]):
            for j in range(self.m[None]):
                v = ti.Vector([-1, -1, -1], dt=ti.i32)
                d = 0.
                if collision_test(i, verts[j][0], verts[j][1], x, dm, v, d) == 1:
                    # print(i, j, verts[j], d, dm, barrier(d, dm))
                    ans += barrier(d, dm) * self.k
        for i in range(self.n[None]):
            for j in range(i):
                xij = x[i] - x[j]
                if xij.norm() > dm: continue
                ans += barrier(xij.norm(), dm) * self.k
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        verts = ti.static(self.vert)
        dm = ti.static(self.d_m)
        for i in range(n):
            for j in range(self.m[None]):
                v = ti.Vector([-1, -1, -1], dt=ti.i32)
                d = 0.
                if collision_test(i, verts[j][0], verts[j][1], x, dm, v, d) == 0: continue
                length = (x[verts[j][0]] - x[verts[j][1]]).norm()
                for k in ti.static(range(3)):
                    dfdd = f_barrier(d, dm) * self.k
                    dddx = ti.Matrix([[0, -1], [1, 0]]) @ (x[v[(k + 1) % 3]] - x[v[k]]) / length
                    f[v[(k + 2) % 3]] += dfdd * dddx
        for i in range(self.n[None]):
            for j in range(i):
                xij = x[i] - x[j]
                if xij.norm() > dm: continue
                dfdd = f_barrier(xij.norm(), dm) * self.k
                dddx = xij.normalized()
                f[i] += dfdd * dddx
                f[j] -= dfdd * dddx
    
    @ti.kernel
    def df(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        verts = ti.static(self.vert)
        dm = ti.static(self.d_m)
        for i in range(n):
            for j in range(self.m[None]):
                v = ti.Vector([-1, -1, -1], dt=ti.i32)
                d = 0.
                if collision_test(i, verts[j][0], verts[j][1], x, dm, v, d) == 0: continue
                length = (x[verts[j][0]] - x[verts[j][1]]).norm()
                for k in ti.static(range(3)):
                    dfdd = f_barrier(d, dm) * self.k
                    dddx = ti.Matrix([[0, -1], [1, 0]]) @ (dx[v[(k + 1) % 3]] - dx[v[k]]) / length
                    f[v[(k + 2) % 3]] += dfdd * dddx
                s = 0.
                for k in ti.static(range(3)):
                    dddx = ti.Matrix([[0, -1], [1, 0]]) @ (x[v[(k + 1) % 3]] - x[v[k]]) / length
                    s += dddx.dot(dx[v[(k + 2) % 3]])
                for k in ti.static(range(3)):
                    dddx = ti.Matrix([[0, -1], [1, 0]]) @ (x[v[(k + 1) % 3]] - x[v[k]]) / length
                    ddf = df_barrier(d, dm) * self.k
                    f[v[(k + 2) % 3]] += ddf * dddx * s
        for i in range(self.n[None]):
            for j in range(i):
                xij = x[i] - x[j]
                d = xij.norm()
                if xij.norm() > dm: continue
                ddf = df_barrier(d, dm) * self.k
                dddx = xij * xij.dot(dx[i] - dx[j])
                tmp = ddf * dddx
                dfdd = f_barrier(d, dm) * self.k
                ddd = (ti.Matrix.identity(ti.f32, 2) - xij.outer_product(xij) / xij.norm_sqr()) @ (dx[i] - dx[j]) / d
                tmp += dfdd * ddd
                f[i] += tmp
                f[j] -= tmp

@ti.data_oriented
class Gravity:
    def __init__(self, args, solver):
        tmp = solver.args.copy()
        tmp.update(args)
        args = tmp

        self.gravity = args['gravity']
        self.solver = solver
    
    def dumps(self):
        return {'class': 'Gravity', 'gravity': self.gravity}
    
    @ti.kernel
    def energy(self, x: ti.template(), n: ti.i32) -> ti.f32:
        ans = 0.
        for i in range(n):
            if self.solver.mass[i] > 1e6: continue
            ans += -x[i].dot(ti.Vector(self.gravity)) * self.solver.mass[i]
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        for i in range(n):
            if self.solver.mass[i] > 1e6: continue
            f[i] += self.gravity * self.solver.mass[i]
    
    @ti.kernel
    def df(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        pass

@ti.data_oriented
class Springs:
    def __init__(self, m_max = 1000):
        self.m = ti.field(dtype=ti.i32, shape=())
        self.m[None] = 0
        self.m_max = m_max
        m = self.m_max
        self.vert = ti.Vector.field(2, dtype=ti.i32, shape=m)
        self.length = ti.field(dtype=ti.f32, shape=m)
        self.k = ti.field(dtype=ti.f32, shape=m)
        self.enable = ti.field(dtype=ti.i32, shape=m)
        self.enable.fill(1)
    
    def add(self, vert, length, k):
        m = self.m[None]
        self.vert[m] = vert
        self.length[m] = length
        self.k[m] = k
        self.m[None] = m + 1
        return m

    def enable(self, i, flag):
        self.enable[i] = flag
    
    @ti.kernel
    def energy(self, x: ti.template(), n: ti.i32) -> ti.f32:
        ans = 0.
        for i in range(self.m[None]):
            if not self.enable[i]: continue
            v = self.vert[i]
            xuv = x[v.x] - x[v.y]
            ans += .5 * self.k[i] * (xuv.norm() - self.length[i])**2
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        for i in range(self.m[None]):
            if not self.enable[i]: continue
            v = self.vert[i]
            xuv = x[v.x] - x[v.y]
            f_tmp = -self.k[i] * (xuv.norm() - self.length[i]) * xuv.normalized()
            f[v.x] += f_tmp
            f[v.y] -= f_tmp

    @ti.kernel
    def df(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        for i in range(self.m[None]):
            if not self.enable[i]: continue
            v = self.vert[i]
            xuv = x[v.x] - x[v.y]
            d = xuv.normalized()
            df = -self.k[i] * (d.outer_product(d))
            if xuv.norm() > self.length[i]:
                df += -self.k[i] * (xuv.norm() - self.length[i]) / xuv.norm() * (ti.Matrix.identity(ti.f32, 2) - d.outer_product(d))
            tmp_df = df @ (dx[v.x] - dx[v.y])
            f[v.x] += tmp_df
            f[v.y] -= tmp_df

@ti.data_oriented
class Attraction:
    def __init__(self, k):
        self.center = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.active = False
        self.k = k
    
    def activate(self, center):
        self.active = True
        self.center[None] = center

    def deactivate(self):
        self.active = False
    
    @ti.kernel
    def energy_k(self, x: ti.template(), n: ti.i32) -> ti.f32:
        ans = 0.
        for i in range(n):
            ans += .5 * self.k * (x[i] - self.center[None]).norm_sqr()
        return ans
    
    def energy(self, x, n):
        if not self.active: 
            return 0
        return self.energy_k(x, n)

    @ti.kernel
    def force_k(self, f: ti.template(), x: ti.template(), n: ti.i32):
        for i in range(n):
            f[i] += -self.k * (x[i] - self.center[None])
    
    def force(self, f, x, n):
        if self.active:
            self.force_k(f, x, n)
    
    @ti.kernel
    def df_k(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        for i in range(n):
            f[i] += -self.k * dx[i]

    def df(self, f, x, dx, n):
        if self.active:
            self.force_k(f, x, n)

# TODO: haven't been tested

@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.func
def Ds(verts, x):
    return ti.Matrix.cols([x[verts[i]] - x[verts[2]] for i in range(2)])
    
@ti.data_oriented
class Elasticity:
    def __init__(self, args, solver):

        Elasticity.arrays = ['vert', 'F_B', 'F_W']
        tmp = solver.args.copy()
        tmp.update(args)
        args = tmp

        self.dim = args['dim']
        k = args['young']
        nu = args['nu']

        self.m = ti.field(dtype=ti.i32, shape=())
        self.m_max = args['m_max']
        self.k = k
        self.nu = nu
        self.mu = k / (2 * (1 + nu))
        self.la = k * nu / ((1 + nu) * (1 - 2 * nu))
        self.vert = ti.Vector.field(self.dim + 1, dtype=ti.i32, shape=self.m_max)
        self.F_B = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.m_max)
        self.F_W = ti.field(dtype=ti.f32, shape=self.m_max)

        for name in Elasticity.arrays:
            if name not in args: continue
            arr = args[name]
            if isinstance(arr, str):
                arr = export.b642np(arr)
            arr = np.array(arr)
            self.__dict__[name].from_numpy(np.resize(arr, (self.m_max, *arr.shape[1:])))
            self.m[None] = arr.shape[0]
    
    def dumps(self):
        ans = {'class': 'Elasticity', 'young': self.k, 'nu': self.nu, 'm_max': self.m_max}
        ans.update([[i, export.np2b64(self.__dict__[i].to_numpy()[:self.m[None]])] for i in Elasticity.arrays])
        return ans
    
    def add(self, vert):
        m = self.m[None]
        self.vert[m] = vert
        self.m[None] = m + 1
        return m
    
    @ti.func
    def Ds(self, verts, x):
        return ti.Matrix.cols([x[verts[i]] - x[verts[self.dim]] for i in range(self.dim)])
    
    @ti.kernel
    def init(self, x: ti.template(), mass: ti.template()):
        for i in range(self.m[None]):
            verts = self.vert[i]
            F = self.Ds(verts, x)
            self.F_B[i] = F.inverse()
            self.F_W[i] = ti.abs(F.determinant()) / ti.static(math.factorial(self.dim))
            for j in ti.static(range(self.dim + 1)):
                mass[verts[j]] += self.F_W[i] / (self.dim + 1)

    
    @ti.kernel
    def energy(self, x: ti.template(), n: ti.i32) -> ti.f32:
        ans = 0.
        for i in range(self.m[None]):
            verts = self.vert[i]
            F = self.Ds(verts, x) @ self.F_B[i]
            I1 = (F.transpose() @ F).trace()
            J = F.determinant()
            ans += self.F_W[i] * (self.mu * (.5 * I1 - 1.5 - ti.log(J)) + self.la * .5 * ti.log(J)**2) # neohookean
            # U, sigma, V = ti.svd(F)
            # s = 0.0
            # for j in ti.static(range(self.dim)):
            #     s += (sigma[j, j] - 1)**2
            # ans += self.F_W[i] * (self.mu * s) # corotated, mu only
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        for i in range(self.m[None]):
            verts = self.vert[i]
            F = self.Ds(verts, x) @ self.F_B[i]
            J = F.determinant()
            P = self.mu * (F - F.inverse().transpose()) + self.la * ti.log(J) * F.inverse().transpose()
            # U, sig, V = ssvd(F)
            # R = U @ V.transpose()
            # P = 2 * self.mu * (F - U @ V.transpose()) # corotated, mu only
            H = -self.F_W[i] * P @ self.F_B[i].transpose()
            for i in ti.static(range(self.dim)):
                f[verts[i]] += H[:, i]
                f[verts[self.dim]] -= H[:, i]

    @ti.kernel
    def df(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        for i in range(self.m[None]):
            verts = self.vert[i]
            F = self.Ds(verts, x) @ self.F_B[i]
            dD = ti.Matrix.cols([dx[verts[j]] - dx[verts[self.dim]] for j in ti.static(range(self.dim))])
            dF = dD @ self.F_B[i]
            Fmt = F.transpose().inverse()
            J = F.determinant()
            dP = self.mu * dF + (self.mu - self.la * ti.log(J)) * Fmt @ dF.transpose() @ Fmt + self.la * (F.inverse() @ dF).trace() * Fmt
            # dP = 2 * self.mu * dF # hacked corotated
            dH = -self.F_W[i] * dP @ self.F_B[i].transpose()
            for i in ti.static(range(self.dim)):
                f[verts[i]] += dH[:, i]
                f[verts[self.dim]] -= dH[:, i]

@ti.data_oriented
class Floor_3d:
    def __init__(self, args, solver):
        self.k = args['k']
        self.solver = solver

    @ti.kernel
    def energy(self, x: ti.template(), n: ti.i32) -> ti.f32:
        ans = .0
        for i in range(n):
            if x[i].y >= 0: continue
            ans += .5 * self.k * x[i].y**2
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        for i in range(n):
            if x[i].y >= 0: continue
            f[i].y += -self.k * x[i].y
    
    @ti.kernel
    def df(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        for i in range(n):
            if x[i].y >= 0: continue
            f[i].y += -self.k * dx[i].y
