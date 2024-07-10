import taichi as ti, numpy as np
import toy
import matplotlib.pyplot as plt
import math
import meshio
import sympy

if __name__ == '__main__':
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

    ti.init(arch=ti.cpu)
    solver = toy.solver.ImplicitSolver()
    solver.target = get_vf_target()
    solver.load_config_file('./config.sh')

    

    res = (1920, 1080)
    window = ti.ui.Window("cloth", res, vsync=True)
    camera = ti.ui.make_camera()
    camera.position(-3.0, 3.0, 3.0)
    camera.lookat(0.5, 0.3, 0.5)
    camera.fov(55)

    canvas = window.get_canvas()
    scene = window.get_scene()
    pause = False
    integration = 'implicit'

    faces = solver.faces
    n_max = solver.n_max
    state = solver

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

    n_stop = -1
    if 'pause' in solver.args: n_stop = 0
    # i_f = 0
    # frames = [state.dumps()]
    gui = window.get_gui()
    # exporter = toy.export.exporter
    while window.running:
        frames = solver.frames
        i_new = gui.slider_int('i_f', solver.i_f, 0, len(frames) - 1)
        i_new = min(i_new, len(frames) - 1)
        solver.load_frame(i_new)
        # if i_new != i_f or flag_change:
        #     state.load_xv(frames[i_new])
        #     i_f = i_new
        #     pause = True
        #     flag_change = False
        # exporter.set_i_f(i_f)
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        for event in window.get_events(ti.ui.PRESS):
            if event.key in [ti.ui.ESCAPE]:
                window.running = False
            if event.key in [' ']:
                if n_stop != 0:
                    n_stop = 0
                else:
                    n_stop = -1
            if event.key in ['n']:
                n_stop = 1
            if event.key in ['o']:
                mesh = meshio.Mesh(state.pos.to_numpy()[:state.n[None]], {'triangle': faces})
                mesh.write('test.ply')
            if event.key in ['j']:
                solver.load_prev_frame()
            if event.key in ['k']:
                solver.load_next_frame()
            if event.key in ['v']:
                fig, ax = plt.subplots()
                X = []
                Y = []
                for i, f in enumerate(frames):
                    X.append(i)
                    tmp = np.array(f['vel'])
                    # print(tmp)
                    Y.append((tmp**2).sum()**.5)
                Y = np.array(Y)
                ax.set_yscale('log')
                ax.plot(X, Y, marker='o')
                plt.subplots_adjust(left=0.2)
                plt.grid(True)
                plt.title('norm of velocity')
                plt.show()
                

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
            # i_f += 1
            # if len(frames) == i_f:
            #     frames.append(None)
            # frames[i_f] = state.dumps()
            if n_stop > 0: n_stop -= 1
        
        # debug_force.fill(0)
        # collision.force(debug_force, state.pos, state.n[None])
        # get_debug_pos()

        # if i_f in exporter.newton:
        #     if 1 in exporter.newton[i_f]:
        #         debug_pos.from_numpy(exporter.newton[i_f][1]['pos'])

        scene.mesh(vertices=state.pos, indices=indices)
        scene.lines(vertices=x_frame, width=1, indices=indices_frame)
        scene.particles(centers=state.pos, radius=0.01, color=(0.6, 0, 0), index_count=state.n[None])
        scene.particles(centers=debug_pos, radius=0.01, color=(0, 0.6, 0), index_count=state.n[None])
        canvas.scene(scene)
        window.show()
        # input()
