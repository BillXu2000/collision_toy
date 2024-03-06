import toy
import taichi as ti, json, numpy as np, os, uuid
from toy import State
from os import path

@ti.kernel
def ax_by(z: ti.template(), a: ti.f32, x: ti.template(), b: ti.f32, y: ti.template()):
    for i in z:
        z[i] = a * x[i] + b * y[i]

@ti.kernel
def set_link(n: ti.i32, target: ti.template(), a: ti.template(), b: ti.template(), verts: ti.template()):
    for i in range(n):
        verts[i] = [i, n + i]
        target[i] = a[i]
        target[n + i] = b[i]

if __name__ == '__main__':

    ti.init(arch=ti.cpu)

    max_n = int(1e3)

    pos = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    dx = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    pos_n = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    target = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    vert = ti.Vector.field(2, dtype=ti.i32, shape=1000)
    tmp = ti.Vector.field(2, dtype=ti.f32, shape=1000)
    tmp.fill(-1)
    n_p = ti.field(ti.i32, shape=())
    pos_linknext = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    vert_linknext = ti.Vector.field(2, dtype=ti.i32, shape=1000)
    window = ti.ui.Window("newton viewer", res=(800, 800), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    canvas.set_background_color((.9,)*3)
    pause = False
    data = {}
    def load(i):
        global data, cur
        fn = f'{i - 1}.log'
        if not path.exists(fn):
            cur -= 1
            return
        with open(fn, 'r') as fi:
            data = json.load(fi)
        pos.from_numpy(np.array(data['pos']))
        dx.from_numpy(np.array(data['dx']))
        ax_by(pos_n, 1, pos, 1, dx)
        set_link(data['n'], pos_linknext, pos, pos_n, vert_linknext)

    with open('spring.log', 'r') as fi:
        vert_data = json.load(fi)
        vert.from_numpy(np.array(vert_data))
    with open('target.log', 'r') as fi:
        target_data = json.load(fi)
        target.from_numpy(np.array(target_data))


    load(1)
    cur = 1
    while window.running:
        canvas.circles(centers=pos, radius=.01, color=(.6,)*3)
        canvas.circles(centers=target, radius=.01, color=(.6, 0, 0))
        canvas.circles(centers=pos_n, radius=.01, color=(0, .6, 0))
        canvas.lines(pos, width=.004, indices=vert)
        canvas.lines(pos_linknext, width=.004, indices=vert_linknext, color=(0, .6, 0))
        # canvas.lines(x_wall, width=.01, indices = i_wall)
        p = pos.to_numpy()
        i = p[4:, 0].argmax() + 4
        gui.text(f'{p[1, 0]} {i} {p[i]}')
        tmp[0] = p[i]
        canvas.circles(centers=tmp, radius=.01, color=(0, 0, .6))
        window.show()
        mouse = window.get_cursor_pos()
        for e in window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                window.running = False
            if e.key == 'n':
                cur += 1
                load(cur)
            if e.key == 'p':
                cur = max(1, cur - 1)
                load(cur)

