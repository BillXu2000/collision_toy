import toy
import taichi as ti, json, numpy as np, os, uuid, subprocess
from toy import State

if __name__ == '__main__':

    ti.init(arch=ti.cpu)

    max_n = int(1e3)

    state = State()
    state.n = ti.field(dtype=ti.i32, shape=())
    state.x = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    state.v = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    state.f = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    state.mass = ti.field(dtype=ti.f32, shape=max_n)
    x0 = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    v0 = ti.Vector.field(2, dtype=ti.f32, shape=max_n)

    state.n[None] = 0
    state.mass.fill(1)
    state.x.fill(-1)

    spring_Y = 10000
    springs = toy.force.Springs()
    attraction = toy.force.Attraction(spring_Y * .01)
    # walls = toy.force.Walls([[0, -1], [0, 1], [-1, 0], [1, 0]], [0, 1, 0, 1], k=spring_Y, d_m=1e-2)
    collision = toy.force.Collision(state.n, springs, k=spring_Y * 1e0, d_m=1e-2)
    elasiticity = toy.force.Elasticity(k=spring_Y)
    # forces = toy.force.Forces([springs, toy.force.Gravity(), attraction, walls, collision])
    # forces = toy.force.Forces([springs, toy.force.Gravity(), attraction, walls])
    # forces = toy.force.Forces([springs, toy.force.Gravity(), attraction])
    # forces = toy.force.Forces([springs, toy.force.Gravity(), attraction, collision])
    forces = toy.force.Forces([elasiticity, toy.force.Gravity(), attraction, collision])
    implicit = toy.force.Implicit(forces, lambda: ti.Vector.field(2, dtype=ti.f32, shape=max_n))
    dt = 1e-2

    def add_polygon(poses):
        n = state.n[None]
        print(n)
        m = len(poses)
        for i in range(m):
            state.x[n + i] = poses[i]
            state.v[n + i] = [0, 0]
            state.mass[n + i] = -1
            if m > 2 or i > 0: springs.add([n + i, n + (i + 1) % m], 0, 0)
        state.n[None] = n + m


    add_polygon([[0.05, 0.05], [1 - 0.05, 0.05], [1 - 0.05, 1 - 0.05], [0.05, 1 - 0.05]])
    # add_polygon([[1, -100], [1, 100]])
    # add_polygon([[0, 0], [0, 1]])

    @ti.kernel
    def advance_explicit():
        n = state.n[None]
        x = ti.static(state.x)
        v = ti.static(state.v)
        f = ti.static(state.f)

        # We use a semi-implicit Euler (aka symplectic Euler) time integrator
        for i in range(n):
            v[i] += dt * f[i] / state.mass[i]

            x[i] += v[i] * dt

            # Collide with four walls
            for d in ti.static(range(2)):
                # d = 0: treating X (horizontal) component
                # d = 1: treating Y (vertical) component

                if x[i][d] < 0:  # Bottom and left
                    x[i][d] = 0  # move particle inside
                    v[i][d] = 0  # stop it from moving further

                if x[i][d] > 1:  # Top and right
                    x[i][d] = 1  # move particle inside
                    v[i][d] = 0  # stop it from moving further
    
    def substep_explicit():
        forces.force(state.f, state.x, state.n[None])
        advance_explicit()

    @ti.kernel
    def advance_implicit(x_1:ti.template()):
        n = state.n[None]
        x = ti.static(state.x)
        v = ti.static(state.v)
        for i in range(n):
            if state.mass[i] != -1:
                v[i] = (x_1[i] - x[i]) / dt
            x[i] = x_1[i]
            # for d in ti.static(range(2)):
            #     if x[i][d] < 0:  # Bottom and left
            #         x[i][d] = 0  # move particle inside
            #         v[i][d] = 0  # stop it from moving further

            #     if x[i][d] > 1:  # Top and right
            #         x[i][d] = 1  # move particle inside
            #         v[i][d] = 0  # stop it from moving further
    
    newton = toy.cg.newton(lambda: ti.Vector.field(2, dtype=ti.f32, shape=max_n))

    def substep_implicit():
        toy.cg.newton.n = state.n
        implicit.set_target(state.x, state.v, state.mass, dt, state.n[None])
        x_1 = newton.newton(implicit.energy, implicit.gradient, implicit.hessian, state.x, collision)
        advance_implicit(x_1)

    def new_particle(pos_x, pos_y):
        u = state.n[None]
        state.x[u] = [pos_x, pos_y]
        state.v[u] = [0, 0]
        state.n[None] = u + 1

        # for v in range(u):
        #     dist = (state.x[u] - state.x[v]).norm()
        #     connection_radius = 0.15
        #     if dist < connection_radius:
        #         # springs.add([u, v], 0.1, spring_Y)
        #         springs.add([u, v], 0.1, 0)
        return u
    
    # for i in range(1, 2):
    #     for j in range(3, 4):
    for i in range(1, 5):
        for j in range(3, 5):
            new_particle(i * .2, j * .2)
            new_particle(i * .2 + .1, j * .2)
            u = new_particle(i * .2, j * .2 + .1)
            springs.add([u, u - 1], 0, 0)
            springs.add([u, u - 2], 0, 0)
            springs.add([u - 1, u - 2], 0, 0)
            elasiticity.add([u - 2, u - 1, u])
    # new_particle(.2, .2)
    elasiticity.init(state.x)


    # gui = ti.GUI("Explicit Mass Spring System", res=(512, 512), background_color=0xDDDDDD)
    window = ti.ui.Window("Taichi MLS-MPM-128", res=(800, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((.9,)*3)
    pause = False
    newton.canvas = canvas
    # x_wall = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    # x_wall[0] = [1, 0]
    # x_wall[1] = [1, 1]
    # x_wall[2] = [0, 1]
    # i_wall = ti.Vector.field(2, dtype=ti.i32, shape=max_n)
    # i_wall[0] = [0, 1]
    # i_wall[1] = [2, 1]

    frames = {}
    frames[0] = state.dumps()
    i_frame = 0
    exporter = toy.export.exporter

    exporter.export({'springs': springs.vert.to_numpy(), 'type': 'springs'})


    while window.running:
        # for i in range(springs.m[None]):
        #     v = springs.vert[i]
        #     if v[0] < state.n[None] and v[1] < state.n[None]:
        #         gui.line(begin=state.x[v[0]], end=state.x[v[1]], radius=2, color=0x444444)
        
        # for i in range(state.n[None]):
        #     c = 0x111111
        #     gui.circle(pos=state.x[i], color=c, radius=5)
        if not pause:
            exporter.set_i_f(i_frame)
            exporter.export(frames[i_frame])
            substep_implicit()
            i_frame += 1
            frames[i_frame] = state.dumps()
        # if window.is_pressed('e'):
        #     toy.cg.ax_by(x0, 1, state.x, 0, state.x)
        #     toy.cg.ax_by(v0, 1, state.v, 0, state.v)
        #     substep_implicit()
        #     toy.cg.ax_by(state.x, 1, x0, 0, state.x)
        #     toy.cg.ax_by(state.v, 1, v0, 0, state.v)
        if not window.is_pressed("v"):
            canvas.circles(centers=state.x, radius=.01, color=(.6,)*3)
            canvas.lines(state.x, width=.004, indices=springs.vert)
        # canvas.lines(x_wall, width=.01, indices = i_wall)
        window.show()
        mouse = window.get_cursor_pos()
        for e in window.get_events(ti.ui.PRESS):
            shift = .5
            if e.key in [ti.ui.ESCAPE]:
                window.running = False
            elif e.key == ti.ui.LMB:
                new_particle(*mouse)
            elif e.key == 'n':
                substep_implicit()
            elif e.key == ' ':
                pause = not pause
            # elif e.key == 'b' and i_frame > 1:
            #     i_frame -= 1
            #     state.loads(frames[i_frame - 1])
            # elif e.key == 's':
            #     # fn = f'./output/{uuid.uuid4().hex}.json'
            #     fn = f'./output/tmp.json'
            #     data = json.dumps(frames[i_frame - 1])
            #     with open(fn, 'w') as fi:
            #         fi.write(data)
            # elif e.key == 'l':
            #     fn = f'./output/tmp.json'
            #     with open(fn, 'r') as fi:
            #         lines = fi.readlines()
            #     j = json.loads('\n'.join(lines))
            #     state.loads(j)
            elif e.key == 'e':
                state.v[1].x -= shift * 10
                state.v[2].x -= shift * 10
            elif e.key == ti.ui.LEFT:
                state.v[1].x -= shift
                state.v[2].x -= shift
            elif e.key == ti.ui.RIGHT:
                state.v[1].x += shift
                state.v[2].x += shift
            elif e.key == ti.ui.DOWN:
                state.v[2].y -= shift
                state.v[3].y -= shift
            elif e.key == ti.ui.UP:
                state.v[2].y += shift
                state.v[3].y += shift
            elif e.key == 'i':
                subprocess.Popen(['python3', './view.py'])
            # elif e.key == 'r':
            #     forces.models.remove(collision)
            # elif e.key == 'c':
            #     if collision not in forces.models: forces.models.append(collision)
        # shift = (walls.b[3] - state.x.to_numpy()[:, 0].max()) / 2
        # state.v[1] = [0, 0]
        # state.v[2] = [0, 0]
        # if window.is_pressed('e'):
        #     state.v[1].x -= shift * 10
        #     state.v[2].x -= shift * 10
        # if window.is_pressed(ti.ui.LEFT):
        # if window.is_pressed(ti.ui.RIGHT):
        #     state.v[1].x += shift
        #     state.v[2].x += shift
        
        # if window.is_pressed(ti.ui.LEFT):
            # walls.b[3] -= shift
            # x_wall[0].x -= shift
            # x_wall[1].x -= shift
        # if window.is_pressed(ti.ui.RIGHT):
        #     shift = max(shift, .005)
        #     walls.b[3] += shift
        #     x_wall[0].x += shift
        #     x_wall[1].x += shift
        # shift = .003
        # shift = (walls.b[1] - state.x.to_numpy()[:, 1].max()) / 2
        # if window.is_pressed(ti.ui.DOWN):
        #     walls.b[1] -= shift
        #     x_wall[1].y -= shift
        #     x_wall[2].y -= shift
        # if window.is_pressed(ti.ui.UP):
        #     shift = max(shift, .005)
        #     walls.b[1] += shift
        #     x_wall[1].y += shift
        #     x_wall[2].y += shift
        if window.is_pressed(ti.ui.RMB):
            attraction.activate(mouse)
        else:
            attraction.deactivate()

