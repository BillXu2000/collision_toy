import taichi as ti
import numpy as np
from . import cg

@ti.data_oriented
class Implicit:
    def __init__(self, forces, gen):
        self.forces = forces
        self.target = gen()
    
    def set_target(self, x, v, mass, dt, n):
        cg.ax_by(self.target, 1, x, dt, v)
        self.mass = mass
        self.dt = dt
        self.n = n
    
    @ti.kernel
    def energy_k(self, x: ti.template(), n: ti.i32) -> ti.f32:
        energy = 0.
        for i in range(n):
            energy += .5 * (x[i] - self.target[i]).norm_sqr() * self.mass[i]
        return energy

    def energy(self, x):
        ans = self.forces.energy(x, self.n)
        return self.energy_k(x, self.n) + self.dt**2 * ans
    
    @ti.kernel
    def gradient_k(self, f: ti.template(), x: ti.template(), n: ti.i32, dt: ti.f32):
        for i in range(n):
            f[i] = (x[i] - self.target[i]) * self.mass[i] - dt**2 * f[i]

    def gradient(self, f, x):
        self.forces.force(f, x, self.n)
        self.gradient_k(f, x, self.n, self.dt)
    
    @ti.kernel
    def hessian_k(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32, dt: ti.f32):
        for i in range(n):
            f[i] = self.mass[i] * dx[i] - dt**2 * f[i]
    
    def hessian(self, f, x, dx):
        self.forces.df(f, x, dx, self.n)
        self.hessian_k(f, x, dx, self.n, self.dt)

class Forces:
    def __init__(self, models=[]):
        self.models = models
    
    def energy(self, x, n):
        ans = 0
        for model in self.models:
            ans += model.energy(x, n)
        return ans

    def force(self, f, x, n):
        f.fill(0)
        for model in self.models:
            model.force(f, x, n)

    def df(self, f, x, dx, n):
        f.fill(0)
        for model in self.models:
            model.df(f, x, dx, n)
    
    def append(self, model):
        self.models.append(model)

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
    def __init__(self, n, springs, k, d_m):
        self.n = n
        self.springs = springs
        self.k = k
        self.d_m = d_m
    
    @ti.kernel
    def ccd(self, x: ti.template(), dx: ti.template()) -> ti.f32:
        alpha = 1.0
        verts = ti.static(self.springs.vert)
        for i in range(self.n[None]):
            for j in range(self.springs.m[None]):
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
        verts = ti.static(self.springs.vert)
        dm = ti.static(self.d_m)
        ans = .0
        for i in range(self.n[None]):
            for j in range(self.springs.m[None]):
                v = ti.Vector([-1, -1, -1], dt=ti.i32)
                d = 0.
                if collision_test(i, verts[j][0], verts[j][1], x, dm, v, d) == 0: continue
                ans += barrier(d, dm) * self.k
        for i in range(self.n[None]):
            for j in range(i):
                xij = x[i] - x[j]
                if xij.norm() > dm: continue
                ans += barrier(xij.norm(), dm) * self.k
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        verts = ti.static(self.springs.vert)
        dm = ti.static(self.d_m)
        for i in range(n):
            for j in range(self.springs.m[None]):
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
        verts = ti.static(self.springs.vert)
        dm = ti.static(self.d_m)
        for i in range(n):
            for j in range(self.springs.m[None]):
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
class Walls:
    def __init__(self, a, b, k, d_m):
        a = np.array(a)
        n = a.shape[0]
        self.n = ti.field(dtype=ti.i32, shape=())
        self.a = ti.Vector.field(a.shape[1], dtype=ti.f32, shape=n)
        self.b = ti.field(dtype=ti.f32, shape=n)
        self.a.from_numpy(a)
        self.b.from_numpy(np.array(b))
        self.n[None] = n
        self.k = k
        self.d_m = d_m

    @ti.kernel
    def energy(self, x: ti.template(), n: ti.i32) -> ti.f32:
        ans = 0.
        inf_flag = 0
        for i in range(n):
            for j in range(self.n[None]):
                d = self.b[j] - self.a[j].dot(x[i])
                if d <= 0: 
                    inf_flag = 1
                if 0 < d < self.d_m:
                    ans += barrier(d, self.d_m) * self.k
        if inf_flag: ans = float('inf')
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        d_m = ti.static(self.d_m)
        for i in range(n):
            for j in range(self.n[None]):
                d = self.b[j] - self.a[j].dot(x[i])
                assert 0 < d
                if 0 < d < self.d_m:
                    dfdd = f_barrier(d, d_m) * self.k
                    f[i] += dfdd * (-self.a[j]) # d(d)/dx
    
    @ti.kernel
    def df(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        d_m = ti.static(self.d_m)
        for i in range(n):
            for j in range(self.n[None]):
                d = self.b[j] - self.a[j].dot(x[i])
                assert 0 < d
                if 0 < d < self.d_m:
                    ddf = df_barrier(d, d_m) * self.k
                    f[i] += ddf * self.a[j] * self.a[j].dot(dx[i])

@ti.data_oriented
class Gravity:
    def __init__(self):
        self.gravity = [0, -9.8]
    
    @ti.kernel
    def energy(self, x: ti.template(), n: ti.i32) -> ti.f32:
        ans = 0.
        for i in range(n):
            ans += -x[i].dot(ti.Vector(self.gravity))
        return ans

    @ti.kernel
    def force(self, f: ti.template(), x: ti.template(), n: ti.i32):
        for i in range(n):
            f[i] += self.gravity
    
    @ti.kernel
    def df(self, f: ti.template(), x: ti.template(), dx: ti.template(), n: ti.i32):
        pass

@ti.data_oriented
class Springs:
    def __init__(self, max_m = 1000):
        self.m = ti.field(dtype=ti.i32, shape=())
        self.m[None] = 0
        self.max_m = max_m
        m = self.max_m
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
@ti.data_oriented
class neohookean:
    def __init__(self, max_m = 1000):
        self.m = ti.field(dtype=ti.i32, shape=())
        self.m[None] = 0
        self.max_m = max_m
        m = self.max_m
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