import taichi as ti, numpy as np
import toy
import matplotlib.pyplot as plt
import math

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
        faces = np.load('./bunny_face.npy').astype(np.float32)
        pos = np.load('./bunny_vert.npy').astype(np.float32)
        tets = np.load('./bunny_ele.npy').astype(np.int32)

        pos[:, 1] += 1.5
        pos *= .3
        vel = np.array(pos)
        vel *= 0
        mass = np.zeros(len(pos), dtype=np.float32)
        # mass += .0001

        volumes = []
        for i in tets:
            print(i)
            volumes.append(np.linalg.det(np.vstack([pos[i[1]] - pos[i[0]], pos[i[2]] - pos[i[0]], pos[i[3]] - pos[i[0]]])) / 6)

        fig, ax = plt.subplots()
        ax.hist([math.log10(i) for i in volumes])
        # ax.hist(volumes)
        # ax.set_xscale('log')
        ax.hist(volumes)
        plt.show()

        exit(0)

    # tets.append(new_tet([0, 1, 0]))
    load_bunny()

    ti.init(arch=ti.cpu)

    n_max = int(1e4)

    dt = 1e-2
    spring_Y = 10000
    args = {'dim': 3, 'float': ti.f32, 'n_max': n_max, 'dt': dt, 'n': len(pos), 'pos': pos, 'vel': vel, 'mass': mass}
    # args.update({'k_collision': spring_Y, 'd_m': 1e-2, 'nu': .49, 'young': spring_Y, 'm_max': n_max, 'forces': [], 'gravity': [0, -9.8, 0]})
    args.update({'k_collision': spring_Y, 'd_m': 1e-2, 'nu': .0, 'young': spring_Y, 'm_max': n_max, 'forces': [], 'gravity': [0, -9.8, 0]})
    state = toy.solver.ImplicitSolver(args)

    # collision = toy.force.Collision({'links': links}, solver=state)
    elasiticity = toy.force.Elasticity_3d({'vert': tets}, solver=state)
    gravity = toy.force.Gravity({'gravity': [0, -9.8, 0]}, solver=state)
    floor = toy.force.Floor_3d({'k': spring_Y}, solver=state)

    forces = [elasiticity, gravity, floor]

    state.forces = forces
    elasiticity.init(state.pos, state.mass)



    res = (1920, 1080)
    window = ti.ui.Window("cloth", res, vsync=True)
    camera = ti.ui.make_camera()
    camera.position(0.5, 1.0, 1.95)
    camera.lookat(0.5, 0.3, 0.5)
    camera.fov(55)

    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    pause = False
    integration = 'implicit'
    collision = True

    indices = ti.field(dtype=ti.i32, shape=(len(faces) * 3))
    indices.from_numpy(np.array(faces).reshape(-1))

    x_frame = ti.Vector.field(3, dtype=ti.f32, shape=(8))
    indices_frame = ti.field(dtype=ti.i32, shape=(12 * 2))
    indices_list = []
    for i in range(8):
        x_frame[i] = [i % 2, (i & 2) >> 1, (i & 4) >> 2]
        for j in [1, 2, 4]:
            if i & j: indices_list.append([i - j, i])
    print(indices_list)
    indices_frame.from_numpy(np.array(indices_list).reshape(-1))


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
                pause = not pause

        # if window.GUI.checkbox('implicit', integration == 'implicit'):
        #     integration = 'implicit'
        # else:
        #     integration = 'explicit'

        # if window.GUI.checkbox('collision', collision):
        #     collision = True
        # else:
        #     collision = False
        
        state.substep()

        scene.mesh(vertices=state.pos, indices=indices)
        scene.lines(vertices=x_frame, width=1, indices=indices_frame)
        canvas.scene(scene)
        window.show()