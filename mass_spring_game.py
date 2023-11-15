# Tutorials (Chinese):
# - https://www.bilibili.com/video/BV1UK4y177iH
# - https://www.bilibili.com/video/BV1DK411A771

import taichi as ti
import toy
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--implicit', action='store_true')
args = parser.parse_args()

ti.init(arch=ti.cpu)

spring_Y = ti.field(dtype=ti.f32, shape=())  # Young's modulus
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())

max_num_particles = 1024
particle_mass = 1.0
dt = 1e-4
substeps = 100
n_wall = 4
d_m = 1e-2

if args.implicit:
    dt *= substeps

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)
wall = ti.Vector.field(2, dtype=ti.f32, shape=n_wall) # (wall.x, wall.y).dot(x) <= wall.z
wall_k = ti.field(dtype=ti.f32, shape=n_wall) # (wall.x, wall.y).dot(x) <= wall.z
attracted = ti.field(dtype=ti.i32, shape=())
attract_xy = ti.Vector.field(2, dtype=ti.f32, shape=())
attracted_k = 0.2
tmp_pos = x

enable_gravity = True
# enable_gravity = False


# rest_length[i, j] == 0 means i and j are NOT connected
rest_length = ti.field(dtype=ti.f32, shape=(max_num_particles, max_num_particles))
newton = toy.cg.newton(lambda: ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles))
toy.cg.newton.n = num_particles

wall.from_numpy(np.array([[0, -1], [0, 1], [-1, 0], [1, 0]]))
wall_k.from_numpy(np.array([0, 1, 0, 1]))

@ti.kernel
def wall_force(f:ti.template(), pos:ti.template()):
    n = num_particles[None]
    for i in range(n):
        for j in range(n_wall):
            d = wall_k[j] - wall[j].dot(pos[i])
            if d >= d_m: continue
            norm = (d_m - d) / d * (2 * d * ti.log(d / d_m) + d - d_m)
            # f[i] += norm * wall[j].dot(pos[i].normalized())
            f[i] += norm * wall[j]
            # print(233, d_m, d, norm, norm * wall[j], f[i])

@ti.kernel
def spring_force(f:ti.template(), x:ti.template()):
    n = num_particles[None]

    # Compute force
    for i in range(n):
        # Gravity
        if enable_gravity:
            f[i] += ti.Vector([0, -9.8]) * particle_mass
        if attracted[None]:
            f[i] += spring_Y[None] * (attract_xy[None] - x[i]) * attracted_k
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()

                # Spring force
                f[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i, j] - 1) * d

                # Dashpot damping
                # v_rel = (v[i] - v[j]).dot(d)
                # f[i] += -dashpot_damping[None] * v_rel * d

@ti.kernel
def advance_explicit():
    n = num_particles[None]

    # We use a semi-implicit Euler (aka symplectic Euler) time integrator
    for i in range(n):
        if not fixed[i]:
            v[i] += dt * f[i] / particle_mass
            # v[i] *= ti.exp(-dt * drag_damping[None])  # Drag damping

            x[i] += v[i] * dt
        else:
            v[i] = ti.Vector([0, 0])

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
    f.fill(0)
    spring_force(f, x)
    advance_explicit()

@ti.kernel
def energy_newton(pos:ti.template()) -> ti.f32:
    n = num_particles[None]
    energy_p = 0.
    inf_flag = False

    # wall
    for i in range(n):
        # gravity
        if enable_gravity:
            energy_p += pos[i].y * 9.8 * particle_mass
        # attract
        if attracted[None]:
            energy_p += .5 * spring_Y[None] * (attract_xy[None] - pos[i]).norm_sqr() * attracted_k
        for j in range(n_wall):
            d = wall_k[j] - wall[j].dot(pos[i])
            if d <= 0: 
                inf_flag = True
            elif d < d_m:
                energy_p += -(d_m - d)**2 * ti.log(d / d_m)
    
    # spring
    for i in range(n):
        for j in range(n):
            if rest_length[i, j] != 0 and i != j:
                x_ij = pos[i] - pos[j]
                energy_p += .5 * spring_Y[None] * (x_ij.norm() - rest_length[i, j])**2
    
    #inertia
    energy_inertia = 0.
    for i in range(n):
        dx = pos[i] - (tmp_pos[i] + dt * v[i])
        energy_inertia += .5 * dx.dot(dx) * particle_mass
    
    # print(energy_inertia, dt**2 * energy_p)
    ans = energy_inertia + dt**2 * energy_p
    if inf_flag:
        ans = float('inf')
    # print(inf_flag, ans)
    return ans

@ti.kernel
def dfdx(ans:ti.template(), pos:ti.template(), dx:ti.template()):
    n = num_particles[None]
    ans.fill(0)
    for i in range(n):
        for j in range(n_wall):
            d = wall_k[j] - wall[j].dot(pos[i])
            if d >= d_m: continue
            w = wall[j].normalized()
            norm = ((d_m/d)**2 + 2 * d_m/d - 2 * ti.log(d/d_m) - 3)
            ans[i] += -norm * w * w.dot(dx[i])
            # print(norm, dx[i], norm * w * w.dot(dx[i]))

    for i in range(n):
        # attract
        if attracted[None]:
            ans[i] += -spring_Y[None] * dx[i] * attracted_k
        for j in range(n):
            if rest_length[i, j] != 0 and i != j:
                x_ij = pos[i] - pos[j]
                d = x_ij.normalized()

                # df = -spring_Y[None] * (x_ij @ x_ij.transpose() * rest_length[i, j] / x_ij.norm()**3)

                df = -spring_Y[None] * (d.outer_product(d))
                if x_ij.norm() > rest_length[i, j]:
                    df += -spring_Y[None] * (x_ij.norm() - rest_length[i, j]) / x_ij.norm() * (ti.Matrix.identity(ti.f32, 2) - d.outer_product(d))
                ans[i] += df @ (dx[i] - dx[j]) / rest_length[i, j]
                # ans[i] += -spring_Y[None] * d * d.dot(dx[i])
    for i in range(n):
        ans[i] = particle_mass * dx[i] - dt**2 * ans[i]
        # ans[i] = particle_mass * dx[i]

@ti.kernel
def newton_gradient(f:ti.template(), x:ti.template()):
    n = num_particles[None]
    for i in range(n):
        f[i] = particle_mass * (x[i] - (tmp_pos[i] + dt * v[i])) - dt**2 * f[i]
        # print(i, particle_mass * (x[i] - (tmp_pos[i] + dt * v[i])), dt**2 * f[i])

def get_force(f, x):
    f.fill(0)
    spring_force(f, x)
    wall_force(f, x)
    newton_gradient(f, x)

@ti.kernel
def advance_implicit(x_1:ti.template()):
    n = num_particles[None]
    for i in range(n):
        v[i] = (x_1[i] - x[i]) / dt
        x[i] = x_1[i]
        # for d in ti.static(range(2)):
        #     if x[i][d] < 0:  # Bottom and left
        #         x[i][d] = 0  # move particle inside
        #         v[i][d] = 0  # stop it from moving further

        #     if x[i][d] > 1:  # Top and right
        #         x[i][d] = 1  # move particle inside
        #         v[i][d] = 0  # stop it from moving further

def substep_implicit():
    x_1 = newton.newton(energy_newton, get_force, dfdx, x)
    advance_implicit(x_1)

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fixed_
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        connection_radius = 0.15
        if dist < connection_radius:
            # Connect the new particle with particle i
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        v[i] += -dt * substeps * (x[i] - p) * 10


def main():
    gui = ti.GUI("Explicit Mass Spring System", res=(512, 512), background_color=0xDDDDDD)

    spring_Y[None] = 1000
    drag_damping[None] = 1
    dashpot_damping[None] = 100

    new_particle(0.3, 0.3, False)
    new_particle(0.3, 0.4, False)
    new_particle(0.4, 0.4, False)

    while True:
        def substep():
            if args.implicit:
                substep_implicit()
            else:
                for step in range(substeps):
                    substep_explicit()
            # print('min', x.to_numpy()[:num_particles[None], 1].min())
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == gui.SPACE:
                paused[None] = not paused[None]
            elif e.key == ti.GUI.LMB:
                new_particle(e.pos[0], e.pos[1], int(gui.is_pressed(ti.GUI.SHIFT)))
            elif e.key == "c":
                num_particles[None] = 0
                rest_length.fill(0)
            elif e.key == "y":
                if gui.is_pressed("Shift"):
                    spring_Y[None] /= 1.1
                else:
                    spring_Y[None] *= 1.1
            elif e.key == "d":
                if gui.is_pressed("Shift"):
                    drag_damping[None] /= 1.1
                else:
                    drag_damping[None] *= 1.1
            elif e.key == "x":
                if gui.is_pressed("Shift"):
                    dashpot_damping[None] /= 1.1
                else:
                    dashpot_damping[None] *= 1.1
            elif e.key == "n":
                substep()

        if gui.is_pressed(ti.GUI.RMB):
            cursor_pos = gui.get_cursor_pos()
            # attract(cursor_pos[0], cursor_pos[1])
            attracted[None] = 1
            attract_xy[None] = [cursor_pos[0], cursor_pos[1]]
        else:
            attracted[None] = 0

        if not paused[None]:
            substep()

        X = x.to_numpy()
        n = num_particles[None]
        # print(X[:n])

        # Draw the springs
        for i in range(n):
            for j in range(i + 1, n):
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x444444)

        # Draw the particles
        for i in range(n):
            c = 0xFF0000 if fixed[i] else 0x111111
            gui.circle(pos=X[i], color=c, radius=5)

        gui.text(
            content="Left click: add mass point (with shift to fix); Right click: attract",
            pos=(0, 0.99),
            color=0x0,
        )
        gui.text(content="C: clear all; Space: pause", pos=(0, 0.95), color=0x0)
        gui.text(
            content=f"Y: Spring Young's modulus {spring_Y[None]:.1f}",
            pos=(0, 0.9),
            color=0x0,
        )
        gui.text(
            content=f"D: Drag damping {drag_damping[None]:.2f}",
            pos=(0, 0.85),
            color=0x0,
        )
        gui.text(
            content=f"X: Dashpot damping {dashpot_damping[None]:.2f}",
            pos=(0, 0.8),
            color=0x0,
        )
        gui.show()


if __name__ == "__main__":
    main()
