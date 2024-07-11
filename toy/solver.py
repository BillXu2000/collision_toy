import taichi as ti
import numpy as np
import json
import meshio
import re
import time
import os
import matplotlib.pyplot as plt
from . import cg, export, force

def load_mesh(fn):
    mesh = meshio.read(fn)
    cells = dict([(i.type, i.data) for i in mesh.cells])
    pos = mesh.points
    faces = cells['triangle']
    tets = cells['tetra']
    return {'points': pos, 'faces': faces, 'tets': tets}

@ti.data_oriented
class ImplicitSolver:
    def __init__(self):
        self.dt = -1
        self.points = []
        self.faces = []
        self.tets = []
        self.fixed = []
        self.forces = []
        self.n_max = -1
        self.dim = -1
        self.args = {}
        self.record_fn = None
        self.frames = []
        self.basic_mass = 0

    
    def insert_frame(self, frame):
        i = frame['i_f']
        while len(self.frames) <= i:
            self.frames.append(None)
        self.frames[i] = frame
    
    def load_log_file(self, fn):
        with open(fn, 'r') as fi:
            lines = fi.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0: continue
            data = json.loads(line)
            self.insert_frame(data)
    
    def load_config_file(self, fn):
        def get_numbers(l, t=float):
            return [t(i) for i in re.findall(r'[+\-\d\.e]+', ','.join(l))]
        with open(fn, 'r') as fi:
            lines = fi.readlines()
        replaces = {}
        for line in lines:
            for i in replaces:
                line = line.replace(i, replaces[i])
            splits = line.split()
            if len(splits) == 0: continue
            if splits[0][0] == '#': continue
            if splits[0][0] == '$':
                replaces[splits[0]] = ' '.join(splits[1:])
            elif splits[0] == 'dim':
                self.dim = int(splits[1])
            elif splits[0] == 'record':
                self.record_fn = splits[1] if len(splits) > 1 else './log/' + '%.3f' % time.time() + '.log'
            elif splits[0] == 'frame':
                self.load_frame(int(splits[1]))
            elif splits[0] == 'mass':
                self.basic_mass = float(splits[1])
            elif splits[0] == 'load':
                assert len(splits) == 2
                self.load_log_file(splits[1])
            elif splits[0] == 'test_gradient':
                self.test_gradient()
            elif splits[0] == 'initialize':
                self.init()
            elif splits[0] == 'exit':
                exit(0)
            elif splits[0] == 'n_max':
                self.n_max = int(splits[1])
            elif splits[0] == 'dt':
                self.dt = float(splits[1])
            elif splits[0] == 'mesh':
                flag_fixed = False
                if splits[1] == 'fixed':
                    del splits[1]
                    flag_fixed = True
                mesh = load_mesh(splits[1])
                if 'faces' in mesh:
                    self.faces.extend((np.array(mesh['faces']) + len(self.points)).tolist())
                if 'tets' in mesh:
                    self.tets.extend((np.array(mesh['tets']) + len(self.points)).tolist())
                points = np.array(mesh['points'])
                if len(splits) > 2:
                    trans = get_numbers(splits[2:], float)
                    if len(trans) <= 3:
                        points = points + np.array(trans)
                    else:
                        m_tmp = round(len(trans)**.5)
                        assert m_tmp**2 == len(trans)
                        trans = np.array(trans).reshape(m_tmp, m_tmp)
                        for i, point in enumerate(points):
                            points[i] = trans @ np.concatenate([point, [1]])
                if flag_fixed:
                    self.fixed.extend(list(range(len(self.points), len(self.points) + len(points))))
                self.points.extend(points.tolist())
            elif splits[0] == 'point':
                p = get_numbers(splits[1:], float)
                self.points.append(p)
            elif splits[0] == 'fixed':
                verts = get_numbers(splits[1:], int)
                self.fixed.extend((np.array(verts) + len(self.points)).tolist())
            elif splits[0] in force.mapping:
                self.forces.append(force.mapping[splits[0]](json.loads(' '.join(splits[1:]))))
            else: # arbitary parameters
                self.args[splits[0]] = json.loads(' '.join(splits[1:])) if len(splits) > 1 else None
    
    def load_frame(self, i):
        if i == self.i_f: return
        if i < 0 or i >= len(self.frames) or self.frames[i] is None: 
            print("frame not found!")
            return
        self.i_f = i
        frame = self.frames[i]
        self.pos.from_numpy(np.resize(frame['pos'], (self.n_max, self.pos.n)))
        self.vel.from_numpy(np.resize(frame['vel'], (self.n_max, self.vel.n)))
        self.mass.from_numpy(np.resize(frame['mass'], (self.n_max,)))
        self.n[None] = frame['n']
        self.dt[None] = frame['dt']
    
    def load_prev_frame(self):
        self.load_frame(self.i_f - 1)
    
    def load_next_frame(self):
        self.load_frame(self.i_f + 1)
    
    def init(self):
        if self.n_max == -1: self.n_max = len(self.points)
        assert self.dim >= 0

        if len(self.frames) > 0: self.points = self.frames[0]['pos']

        def to_field(v, type):
            ans = ti.field(type, shape=())
            ans[None] = v
            return ans
        self.dt = to_field(self.dt, ti.f32)
        self.n = to_field(len(self.points), ti.i32)

        density = 1e3 # TODO: variable density
        self.mass = [self.basic_mass] * len(self.points)
        for tet in self.tets:
            p = np.array(self.points)[tet]
            v = np.linalg.det(p[1:] - p[0]) / 6
            for i in range(4):
                self.mass[tet[i]] += v * density
        for i in self.fixed:
            self.mass[i] = -1

        self.gen_field = lambda: ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_max)
        def gen(np_arr):
            ans = self.gen_field()
            assert len(np_arr.shape) == 2
            assert np_arr.shape[1] == ans.n
            np_arr = np.resize(np_arr, ans.shape + (ans.n,))
            ans.from_numpy(np_arr)
            return ans
        
        self.pos = gen(np.array(self.points))
        self.vel = gen(np.array([[0.] * self.dim]))
        mass_np = self.mass
        self.mass = ti.field(ti.f32, shape=self.n_max)
        self.mass.from_numpy(np.resize(mass_np, self.n_max))

        self.x_tmp = self.gen_field()
        self.df_tmp = self.gen_field()

        self.newton = cg.newton(self.n, self.gen_field)
        self.ans = self.newton.pos

        for force in self.forces:
            force.init(self)
        
        self.i_f = 0
        self.insert_frame(self.dumps())
        if self.record_fn is not None:
            dir = os.path.dirname(self.record_fn)
            os.makedirs(dir, exist_ok=True)
            with open(self.record_fn, 'w') as fi:
                pass
            self.export(self.frames[0])


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
        ans['i_f'] = self.i_f
        vars = ['n', 'dt']
        ans.update([[i, self.__dict__[i][None]] for i in vars])
        # constants = ['dim', 'n_max']
        # ans.update([[i, self.__dict__[i]] for i in constants])
        arrays = ['pos', 'vel', 'mass']
        # ans.update([[i, export.np2b64(self.__dict__[i].to_numpy()[:self.n[None]])] for i in arrays])
        ans.update([[i, self.__dict__[i].to_numpy()[:self.n[None]].tolist()] for i in arrays])
        # ans['forces'] = []
        # for i in self.forces:
        #     ans['forces'].append(i.dumps())
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
    
    def export(self, data):
        with open(self.record_fn, 'a') as fi:
            fi.write(json.dumps(data) + '\n')

    def substep(self):
        self.run()
        self.advance()
        self.i_f += 1
        data = self.dumps()
        self.insert_frame(data)
        if self.record_fn is not None: self.export(data)
    
    def ccd(self, pos, dx):
        ans = 1.0
        for force in self.forces:
            if hasattr(force, 'ccd'):
                ans = min(ans, force.ccd(pos, dx))
        return ans
    
    def run(self):
        self.newton.newton(self.energy, self.gradient, self.hessian, self.pos, self.ccd)
    
    def test_gradient(self):
        self.x_tmp.copy_from(self.pos)
        e0 = self.energy(self.x_tmp)
        self.gradient(self.df_tmp, self.x_tmp)
        df = self.df_tmp.to_numpy()
        x0 = self.pos.to_numpy()
        rng = np.random.default_rng()
        delta = rng.uniform(-1, 1, x0.shape)
        for i in range(self.n[None]):
            if self.mass[i] == -1: delta[i] = 0
        delta /= (delta**2).sum()**.5

        fig, ax = plt.subplots()
        X = []
        Y = []
        for i in range(30):
            step = 2**-i
            self.x_tmp.from_numpy(x0 + delta * step)
            ei = self.energy(self.x_tmp)
            # print(i, ei - (e0 + (df * delta * step).sum()))
            ref = (df * delta).sum()
            err = abs((ei - e0) / step - ref)
            # print(i, err / e0, err, (df * delta).sum(), ei - e0, e0)
            print(i, abs(err / ref), err, (df * delta).sum(), ei, e0)
            X.append(step)
            Y.append(abs(err / ref))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(X, Y, marker='o')
        plt.subplots_adjust(left=0.2)
        plt.grid(True)
        plt.title('finite diffrence')
        plt.show()

    @ti.kernel
    def energy_k(self, x: ti.template()) -> ti.f32:
        energy = 0.
        n = self.n[None]
        dt = self.dt[None]
        for i in range(n):
            if self.mass[i] == -1: continue
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
            if self.mass[i] == -1: 
                de[i] = 0
                continue
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
            if self.mass[i] == -1: 
                dde[i] = 0
                continue
            dde[i] = self.mass[i] * dx[i] - dt**2 * dde[i]
            # dde[i] = self.mass[i] * dx[i] # FIXME: hack
    
    def hessian(self, dde, x, dx):
        dde.fill(0)
        for force in self.forces:
            force.df(dde, x, dx, self.n[None])
        self.hessian_k(dde, x, dx)