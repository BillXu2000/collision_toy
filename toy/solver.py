import taichi as ti
import numpy as np
import json
from . import cg, export, force

@ti.data_oriented
class ImplicitSolver:
    def __init__(self, args):
        self.args = args
        for i in args:
            k = args[i]
            if isinstance(k, str):
                args[i] = export.b642np(k)
        self.dim = args['dim']
        self.n_max = args['n_max']

        self.dt = ti.field(ti.f32, shape=())
        if 'dt' in args:
            self.dt[None] = args['dt']
        self.n = ti.field(ti.i32, shape=())
        if 'n' in args:
            self.n[None] = args['n']
        self.gen_field = lambda: ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_max)

        def gen(name):
            ans = self.gen_field()
            if name in args:
                np_arr = np.array(args[name])
                assert len(np_arr.shape) == 2
                assert np_arr.shape[1] == ans.n
                np_arr = np.resize(np_arr, ans.shape + (ans.n,))
                # np_arr.resize((ans.shape + (ans.n,)))
                ans.from_numpy(np_arr)
            return ans
        self.pos = gen('pos')
        self.vel = gen('vel')
        self.mass = ti.field(ti.f32, shape=self.n_max)
        if 'mass' in args:
            np_arr = np.array(args['mass'])
            assert len(np_arr.shape) == 1
            self.mass.from_numpy(np.resize(np_arr, self.mass.shape))

        self.newton = cg.newton(self.n, self.gen_field)
        self.ans = self.newton.pos
        self.forces = []

        for i in args['forces']:
            self.forces.append(force.loads(i, self))
    
    def add_forces(self, forces):
        self.forces += forces
    
    def load_xv(self, args):
        def load(name):
            field = self.__dict__[name]
            if name in args:
                np_arr = args[name]
                if isinstance(np_arr, str):
                    np_arr = export.b642np(np_arr)
                np_arr = np.array(np_arr)
                assert len(np_arr.shape) == 2
                assert np_arr.shape[1] == self.dim
                np_arr.resize((field.shape + (field.n,)))
                field.from_numpy(np_arr)
        load('pos')
        load('vel')
    
    def dumps(self):
        ans = {}
        constants = ['dim', 'n_max']
        ans.update([[i, self.__dict__[i]] for i in constants])
        arrays = ['pos', 'vel', 'mass']
        ans.update([[i, export.np2b64(self.__dict__[i].to_numpy()[:self.n[None]])] for i in arrays])
        vars = ['dt', 'n']
        ans.update([[i, self.__dict__[i][None]] for i in vars])
        ans['forces'] = []
        for i in self.forces:
            ans['forces'].append(i.dumps())
        return ans
    
    def update(self, args):
        for name in args:
            if name == 'forces':
                self.forces = args['forces']
            else:
                self.__dict__[name].from_numpy(np.array(args[name]))
    
    @ti.kernel
    def advance(self):
        n = self.n[None]
        x = ti.static(self.pos)
        v = ti.static(self.vel)
        dt = self.dt[None]
        for i in range(n):
            if self.mass[i] < 5e4:
                v[i] = (self.ans[i] - x[i]) / dt
            x[i] = self.ans[i]

    def substep(self):
        self.run()
        self.advance()
    
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
            target = self.pos[i] + dt * self.vel[i]
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
            target = self.pos[i] + dt * self.vel[i]
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