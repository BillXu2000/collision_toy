import taichi as ti
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
    def __init__(self):
        self.m = ti.field(dtype=ti.i32, shape=())
        self.m[None] = 0
        self.max_m = int(1e5)
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