import taichi as ti, numpy as np
import toy
import matplotlib.pyplot as plt
import math
import meshio
import sympy

if __name__ == '__main__':
    global vel
    pos = []
    vel = []
    mass = []
    tets = []
    faces = []
    # links = []
    # triangles = []

    def new_particle(p):
        pos.append(p)
        vel.append([0, 0, 0])
        mass.append(1)
        return len(pos) - 1
    
    def new_tet(pos):
        vert = []
        for i in range(4):
            tmp = np.array(pos, dtype=np.float32)
            if i < 3:
                tmp[i] += .3
            vert.append(new_particle(tmp))
        for i in range(4):
            faces.append(vert[:i] + vert[i + 1:])
        return vert
    
    def load_bunny():
        global faces, tets, pos, mass, vel
        # faces = np.load('./bunny_face.npy').astype(np.float32)
        # pos = np.load('./bunny_vert.npy').astype(np.float32)
        # tets = np.load('./bunny_ele.npy').astype(np.int32)

        # mesh = meshio.read('./tet.msh')
        # mesh = meshio.read('./oct.msh')
        # mesh = meshio.read('./scratch/bunny1k.msh')
        mesh = meshio.read('./scratch/new.msh')
        cells = dict([(i.type, i.data) for i in mesh.cells])
        pos = mesh.points
        faces = cells['triangle']
        tets = cells['tetra'][:]



        def add_mesh(offset):
            global faces, tets, pos
            faces = np.array(list(faces) + list(cells['triangle'] + len(pos)))
            tets = np.array(list(tets) + list(cells['tetra'][:] + len(pos)))
            pos = np.array(list(pos) + list(mesh.points + offset))
        
        add_mesh([.9, 1.5, .9])
        # add_mesh([.8, 2.6, .8])

        pos[:, 1] -= pos[:, 1].min()
        pos *= .3
        vel = np.array(pos)
        vel *= 0
        mass = np.zeros(len(pos), dtype=np.float32)

        # volumes = []
        # for i in tets:
        #     volumes.append(np.linalg.det(np.vstack([pos[i[1]] - pos[i[0]], pos[i[2]] - pos[i[0]], pos[i[3]] - pos[i[0]]])) / 6)
        # print('volume sum', sum(volumes))
        

        # fig, ax = plt.subplots()
        # ax.hist([math.log10(i) for i in volumes])
        # # ax.hist(volumes)
        # # ax.set_xscale('log')
        # ax.hist(volumes)
        # plt.show()

        # exit(0)

    # tets.append(new_tet([0, 1, 0]))
    load_bunny()
    edges = set()
    for i in tets:
        for x in range(4):
            for y in range(x + 1, 4):
                edges.add(tuple(sorted((i[x], i[y]))))
    
    def get_spring_target():
        def norm2(x):
            return sum([i**2 for i in x])**0.5
        def vector_sympy(name, n):
            return [sympy.Symbol(name % i) for i in range(n)]
        x = vector_sympy('x[%d]', 3)
        k = sympy.Symbol('data.k')
        l_0 = sympy.Symbol('data.l_0[i]')
        target = k / 2 * (norm2(x) - l_0)**2 * l_0
        st = toy.spm.target2ti(target)
        return toy.string2module(st)
    
    def get_vf_target():
        def vector_sympy(name, n):
            return sympy.Matrix([sympy.Symbol(name % i) for i in range(n)])
        x = vector_sympy('x[0, %d]', 3)
        y = vector_sympy('x[1, %d]', 3)
        z = vector_sympy('x[2, %d]', 3)
        target = x.dot(y.cross(z))
        st = toy.spm.target2ti(target)
        return toy.string2module(st)

    vol_tot = 0.0
    for tet in tets:
        diff = [pos[tet[i]] - pos[tet[0]] for i in range(1, 4)]
        volume = np.linalg.det(np.array(diff)) / 6
        vol_tot += volume

    ti.init(arch=ti.cpu)

    n_max = int(1e4)

    dt = 1e-2
    young = 1e4
    density = 1000
    args = {'dim': 3, 'float': ti.f32, 'n_max': n_max, 'dt': dt, 'n': len(pos), 'pos': pos, 'vel': vel, 'mass': mass}
    args.update({'k_collision': young, 'd_m': 1e-2, 'nu': .4, 'young': young, 'm_max': n_max, 'forces': [], 'gravity': [0, -9.8, 0]})
    # args.update({'k_collision': spring_Y, 'd_m': 1e-2, 'nu': .0, 'young': spring_Y, 'm_max': n_max, 'forces': [], 'gravity': [0, -9.8, 0]})
    state = toy.solver.ImplicitSolver(args)
    # print(args['n'])

    # collision = toy.force.Collision({'links': links}, solver=state)
    elasiticity = toy.force.Elasticity({'vert': tets}, solver=state)
    gravity = toy.force.Gravity({'gravity': [0, -9.8, 0]}, solver=state)
    floor = toy.force.Floor_3d({'k': young}, solver=state)
    spring = toy.force.Spring_sympy({'vert': edges, 'target': get_spring_target()}, solver=state)
    collision = toy.force.Collision_sympy({'faces': faces, 'target': get_vf_target()}, solver=state)


    # state.add_forces([elasiticity, gravity, floor])
    state.add_forces([spring, gravity, floor, collision])
    elasiticity.init(state.pos, state.mass)

    spring.init(state.pos)

    state.mass.fill(1 / len(pos)) # TODO : mass hack



    res = (1920, 1080)
    window = ti.ui.Window("cloth", res, vsync=True)
    camera = ti.ui.make_camera()
    camera.position(0.5, 1.0, 1.95)
    camera.lookat(0.5, 0.3, 0.5)
    camera.fov(55)

    canvas = window.get_canvas()
    scene = window.get_scene()
    pause = False
    integration = 'implicit'

    indices = ti.field(dtype=ti.i32, shape=(len(faces) * 3))
    indices.from_numpy(np.array(faces).reshape(-1))

    x_frame = ti.Vector.field(3, dtype=ti.f32, shape=(8))
    indices_frame = ti.field(dtype=ti.i32, shape=(12 * 2))
    indices_list = []
    for i in range(8):
        x_frame[i] = [i % 2, (i & 2) >> 1, (i & 4) >> 2]
        for j in [1, 2, 4]:
            if i & j: indices_list.append([i - j, i])
    # print(indices_list)
    indices_frame.from_numpy(np.array(indices_list).reshape(-1))

    debug_force = ti.Vector.field(3, dtype=ti.f32, shape=(n_max))
    debug_pos = ti.Vector.field(3, dtype=ti.f32, shape=(n_max))

    @ti.kernel
    def get_debug_pos():
        for i in range(state.n[None]):
            debug_pos[i] = state.pos[i] + debug_force[i] / state.mass[i] * state.dt[None]

    n_stop = 1
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        for event in window.get_events(ti.ui.PRESS):
            if event.key in [ti.ui.ESCAPE]:
                window.running = False
            if event.key in [' ']:
                n_stop = -n_stop - 1
            if event.key in ['n']:
                n_stop = 1
        
        # if window.GUI.checkbox('implicit', integration == 'implicit'):
        #     integration = 'implicit'
        # else:
        #     integration = 'explicit'

        # if window.GUI.checkbox('collision', collision):
        #     collision = True
        # else:
        #     collision = False

        if n_stop != 0:
            state.substep()
            if n_stop > 0: n_stop -= 1
        
        debug_force.fill(0)
        collision.force(debug_force, state.pos, state.n[None])
        get_debug_pos()

        scene.mesh(vertices=state.pos, indices=indices)
        scene.lines(vertices=x_frame, width=1, indices=indices_frame)
        scene.particles(centers=state.pos, radius=0.01, color=(0.6, 0, 0), index_count=state.n[None])
        scene.particles(centers=debug_pos, radius=0.01, color=(0, 0.6, 0), index_count=state.n[None])
        canvas.scene(scene)
        window.show()
        # input()
