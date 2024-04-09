import taichi as ti
import numpy as np
import json
from . import cg, export

@ti.data_oriented
class ImplicitSolver:
    def __init__(self, args):
        # self.dim = args.get('dim', 2)
        # self.float = args.get('float', ti.f32)
        # self.n_max = args.get('n_max', 1000)
        self.dim = args['dim']
        self.float = args['float']
        self.n_max = args['n_max']

        self.dt = ti.field(self.float, shape=())
        if 'dt' in args:
            self.dt[None] = args['dt']
        self.n = ti.field(ti.i32, shape=())
        if 'n' in args:
            self.n[None] = args['n']
        self.gen_field = lambda: ti.Vector.field(self.dim, dtype=self.float, shape=self.n_max)

        def gen(name):
            ans = self.gen_field()
            if name in args:
                ans.from_numpy(args[name])
            return ans
        self.pos = gen('pos')
        self.vel = gen('vel')
        self.mass = ti.field(self.float, shape=self.n_max)
        if 'mass' in args:
            self.mass.from_numpy(args['mass'])

        self.newton = cg.newton(self.n, self.gen_field)
        self.ans = self.newton.pos
        
    def update(self, args):
        for name in args:
            if name == 'forces':
                self.forces = args['forces']
            else:
                self.__dict__[name].from_numpy(np.array(args[name]))
    
    def ccd(self, pos, dx):
        ans = 1.0
        for force in self.forces:
            if hasattr(force, 'ccd'):
                ans = min(ans, force.ccd(pos, dx))
        return ans
    
    def run(self):
        self.newton.newton(self.energy, self.gradient, self.hessian, self.pos, self.ccd)

    @ti.kernel
    def energy_k(self, x: ti.template()) -> ti.f32:
        energy = 0.
        n = self.n[None]
        dt = self.dt[None]
        for i in range(n):
            target = self.x[i] + dt * self.v[i]
            energy += .5 * (x[i] - target).norm_sqr() * self.mass[i]
        return energy

    def energy(self, x):
        ans = 0.
        for force in self.forces:
            ans += force.energy(x, self.n[None])
        ans_k = self.energy_k(x)
        return ans_k + self.dt[None]**2 * ans
    
    @ti.kernel
    def gradient_k(self, de: ti.template(), x: ti.template()):
        n = self.n[None]
        dt = self.dt[None]
        for i in range(n):
            target = self.x[i] + dt * self.v[i]
            de[i] = (x[i] - target) * self.mass[i] - dt**2 * de[i]

    def gradient(self, de, x):
        de.fill(0)
        for force in self.forces:
            force.force(de, x, self.n[None])
        self.gradient_k(de, x)
    
    @ti.kernel
    def hessian_k(self, dde: ti.template(), x: ti.template(), dx: ti.template()):
        n = self.n[None]
        dt = self.dt[None]
        for i in range(n):
            dde[i] = self.mass[i] * dx[i] - dt**2 * dde[i]
    
    def hessian(self, dde, x, dx):
        dde.fill(0)
        for force in self.forces:
            force.df(dde, x, dx, self.n[None])
        self.hessian_k(dde, x, dx)