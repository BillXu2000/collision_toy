import toy
import taichi as ti, json, numpy as np, os, uuid, subprocess
from toy import State
import cProfile
import sys

if __name__ == '__main__':

    pos = []
    vel = []
    mass = []
    links = []
    triangles = []

    def add_polygon(poses):
        n = len(pos)
        m = len(poses)
        for i in range(m):
            pos.append(poses[i])
            vel.append([0, 0])
            mass.append(1e8)
            if m > 2 or i > 0: links.append([n + i, n + (i + 1) % m])

    add_polygon([[0.05, 0.05], [1 - 0.05, 0.05], [1 - 0.05, 1 - 0.05], [0.05, 1 - 0.05]])

    def new_particle(pos_x, pos_y):
        pos.append([pos_x, pos_y])
        vel.append([0, 0])
        mass.append(1)
        return len(pos) - 1
    
    for i in range(1, 5):
        for j in range(3, 5):
            new_particle(i * .2, j * .2)
            new_particle(i * .2 + .1, j * .2)
            u = new_particle(i * .2, j * .2 + .1)
            links.append([u - 2, u - 1])
            links.append([u, u - 1])
            links.append([u, u - 2])
            triangles.append([u - 2, u - 1, u])

    ti.init(arch=ti.cpu)

    n_max = int(1e3)

    dt = 1e-2
    spring_Y = 10000
    args = {'dim': 2, 'float': ti.f32, 'n_max': n_max, 'dt': dt, 'n': len(pos), 'pos': pos, 'vel': vel, 'mass': mass}
    # args.update({'forces': [{'class': 'Gravity'}, {'class': 'Collision'}, {'class': 'gravity'},]})
    args.update({'k_collision': spring_Y, 'd_m': 1e-2, 'nu': .49, 'young': spring_Y, 'm_max': n_max, 'forces': [], 'gravity': [0, -9.8]})
    state = toy.solver.ImplicitSolver(args)

    collision = toy.force.Collision({'links': links}, solver=state)
    elasiticity = toy.force.Elasticity({'vert': triangles}, solver=state)

    forces = [elasiticity, toy.force.Gravity({}, state), collision]

    state.forces = forces
    elasiticity.init(state.pos)

    dump = state.dumps()
    with open('init.log', 'w') as fi:
        json.dump(dump, fi)
