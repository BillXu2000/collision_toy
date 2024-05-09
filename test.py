import base64
import numpy as np
import io
import wildmeshing as wm
import matplotlib.pyplot as plt
import math
import meshio
import pymeshlab

def updated_surface(vf, vt, f):
    hash_vf = {}
    for i, u in enumerate(vf):
        hash_vf[tuple(u.tolist())] = i
    assert len(vf) == len(hash_vf)
    mapping = [-1] * len(vf)
    for i, u in enumerate(vt):
        t = tuple(u.tolist())
        if t in hash_vf:
            mapping[hash_vf[t]] = i
    assert min(mapping) != -1
    print(mapping)
    ans = f.copy()
    for i in range(len(f)):
        for j in range(len(f[i])):
            ans[i, j] = mapping[f[i, j]]
    return ans

def mesh2tet():
    ml_ms = pymeshlab.MeshSet()
    ml_ms.load_new_mesh('./bunny1k.ply')
    mesh = ml_ms.current_mesh()
    v = mesh.vertex_matrix()
    f = mesh.face_matrix()

    tetra = wm.Tetrahedralizer(stop_quality=1000)
    tetra.set_mesh(v, f)
    tetra.tetrahedralize()
    print(dir(tetra))
    vt, tt = tetra.get_tet_mesh()
    ft = updated_surface(v, vt, f)
    print(vt, ft)

    # ans = pymeshlab.Mesh(vertex_matrix=vt, face_matrix=ft)
    # _ms = pymeshlab.MeshSet()
    # _ms.add_mesh(ans)
    # _ms.save_current_mesh('tetra.ply')

    msh = meshio.Mesh(vt, {'tetra': tt, 'triangle': ft})
    msh.write('test.msh')

    face_all = []
    for i in tt:
        for j in range(4):
            face_all.append([i[k] for k in range(4) if k != j])
    
    msh_all = meshio.Mesh(vt, {'tetra': tt, 'triangle': face_all})
    msh_all.write('all.ply')
    # msh.write('test.ply')

    msh = meshio.read('./test.msh')
    dict = {}
    for i in msh.cells:
        dict[i.type] = i.data
    print(dict)

    # volumes = []
    # for i in tt:
    #     volumes.append(np.linalg.det(np.vstack([vt[i[1]] - vt[i[0]], vt[i[2]] - vt[i[0]], vt[i[3]] - vt[i[0]]])) / 6)

    # fig, ax = plt.subplots()
    # ax.hist([math.log10(i) for i in volumes])
    # ax.hist(volumes)
    # plt.show()

def test_export():
    import toy
    # t = np.arange(25, dtype=np.float64)
    t = np.array([['233', 2] + [0]*10000, [3, 4] + [0]*10000])
    b64 = toy.export.np2b64(t)
    q = toy.export.b642np(b64)
    # s = base64.b64encode(t)
    # r = base64.decodebytes(s)
    # q = np.frombuffer(r, dtype=np.float64)

    print(q)
    print(t)
    print(b64)
    print('instance', isinstance(q, np.ndarray))
    # print(np.allclose(q, t))

def test_taichi():
    import taichi as ti
    ti.init(arch=ti.cpu)
    a = ti.Vector.field(2, ti.i32, shape=(2))
    print(a, a.shape)
    a.from_numpy(np.array([[1, 2], [3, 4]]))
    print(a, a.n, a.m)
    print(a.tolist())

# mesh = meshio.read('bunny1k.msh')
# print(dir(mesh))
# print(mesh.points)
# print(dir(mesh.cells[0]))
# print(mesh.cells[0].data)
# print(mesh.faces)

# mesh2tet()

# msh = meshio.read('./tet2.msh')
# tt = msh.cells[0].data
# face_all = []
# for i in tt:
#     for j in range(4):
#         face_all.append([i[k] for k in range(4) if k != j])

# msh_all = meshio.Mesh(msh.points, {'triangle': face_all})
# msh_all.write('all.ply')
# dict = {}
# for i in msh.cells:
#     print(i)
#     dict[i.type] = i.data
# print(dict)

from toy import tools
tools.mesh2tet('bunny1k.ply', 'bunny1k.msh')