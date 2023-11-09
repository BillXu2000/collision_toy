import toy
import taichi as ti

if __name__ == '__main__':

    ti.init(arch=ti.cpu)

    max_n = int(1e5)

    state = lambda: state
    state.n = ti.field(dtype=ti.i32, shape=())
    state.x = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    state.v = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    state.f = ti.Vector.field(2, dtype=ti.f32, shape=max_n)
    state.mass = ti.field(dtype=ti.f32, shape=max_n)

    state.n[None] = 0
    state.mass.fill(1)

    springs = toy.force.Springs()
    forces = toy.force.Forces([springs, toy.force.Gravity()])
    implicit = toy.force.Implicit(forces, lambda: ti.Vector.field(2, dtype=ti.f32, shape=max_n))
    spring_Y = 10000
    dt = 1e-2

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
            v[i] = (x_1[i] - x[i]) / dt
            x[i] = x_1[i]
            for d in ti.static(range(2)):
                if x[i][d] < 0:  # Bottom and left
                    x[i][d] = 0  # move particle inside
                    v[i][d] = 0  # stop it from moving further

                if x[i][d] > 1:  # Top and right
                    x[i][d] = 1  # move particle inside
                    v[i][d] = 0  # stop it from moving further
    
    newton = toy.cg.newton(lambda: ti.Vector.field(2, dtype=ti.f32, shape=max_n))

    def substep_implicit():
        implicit.set_target(state.x, state.v, state.mass, dt, state.n[None])
        x_1 = newton.newton(implicit.energy, implicit.gradient, implicit.hessian, state.x)
        advance_implicit(x_1)

    def new_particle(pos_x, pos_y):
        u = state.n[None]
        state.x[u] = [pos_x, pos_y]
        state.v[u] = [0, 0]
        state.n[None] = u + 1

        for v in range(u):
            dist = (state.x[u] - state.x[v]).norm()
            connection_radius = 0.15
            if dist < connection_radius:
                springs.add([u, v], 0.1, spring_Y)
    
    new_particle(.3, .3)
    new_particle(.3, .4)
    new_particle(.4, .4)

    gui = ti.GUI("Explicit Mass Spring System", res=(512, 512), background_color=0xDDDDDD)

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False

        for i in range(springs.m[None]):
            v = springs.vert[i]
            if v[0] < state.n[None] and v[1] < state.n[None]:
                gui.line(begin=state.x[v[0]], end=state.x[v[1]], radius=2, color=0x444444)
        
        for i in range(state.n[None]):
            c = 0x111111
            gui.circle(pos=state.x[i], color=c, radius=5)
        substep_implicit()
        gui.show()
