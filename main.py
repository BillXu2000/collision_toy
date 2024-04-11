import toy
import taichi as ti, json, numpy as np, os, uuid, subprocess
from toy import State
import cProfile

if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    with open('./init.log', 'r') as fi:
        args = json.load(fi)
        state = toy.solver.ImplicitSolver(args)
    for i in state.forces:
        if isinstance(i, toy.force.Collision):
            collision = i

    state.x = state.pos
    state.v = state.vel


    # gui = ti.GUI("Explicit Mass Spring System", res=(512, 512), background_color=0xDDDDDD)
    window = ti.ui.Window("Taichi MLS-MPM-128", res=(800, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((.9,)*3)

    # frames = {}
    # frames[0] = state.dumps()
    # i_frame = 0
    exporter = toy.export.exporter

    # exporter.export({'springs': springs.vert.to_numpy(), 'type': 'springs'})
    exporter.export({'springs': collision.vert.to_numpy(), 'type': 'springs'})

    pause = False

    while window.running:
        # for i in range(springs.m[None]):
        #     v = springs.vert[i]
        #     if v[0] < state.n[None] and v[1] < state.n[None]:
        #         gui.line(begin=state.x[v[0]], end=state.x[v[1]], radius=2, color=0x444444)
        
        # for i in range(state.n[None]):
        #     c = 0x111111
        #     gui.circle(pos=state.x[i], color=c, radius=5)
        if not pause:
            # exporter.set_i_f(i_frame)
            # exporter.export(frames[i_frame])
            # substep_implicit()
            state.substep()
            # i_frame += 1
            # frames[i_frame] = state.dumps()
        # if window.is_pressed('e'):
        #     toy.cg.ax_by(x0, 1, state.x, 0, state.x)
        #     toy.cg.ax_by(v0, 1, state.v, 0, state.v)
        #     substep_implicit()
        #     toy.cg.ax_by(state.x, 1, x0, 0, state.x)
        #     toy.cg.ax_by(state.v, 1, v0, 0, state.v)
        if not window.is_pressed("v"):
            canvas.circles(centers=state.x, radius=.01, color=(.6,)*3)
            canvas.lines(state.x, width=.004, indices=collision.vert)
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
                state.substep()
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

        # if window.is_pressed(ti.ui.RMB):
        #     attraction.activate(mouse)
        # else:
        #     attraction.deactivate()

