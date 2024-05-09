
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

def mesh2tet(input, output):
    import meshio
    import wildmeshing as wm
    mesh = meshio.read(input)
    v = mesh.points
    f = mesh.cells[0].data

    tetra = wm.Tetrahedralizer(stop_quality=1000)
    tetra.set_mesh(v, f)
    tetra.tetrahedralize()
    # print(dir(tetra))
    vt, tt = tetra.get_tet_mesh()
    ft = updated_surface(v, vt, f)
    # print(vt, ft, tt)

    msh = meshio.Mesh(vt, {'tetra': tt, 'triangle': ft})
    msh.write(output)
    face_all = []
    for i in tt:
        for j in range(4):
            face_all.append([i[k] for k in range(4) if k != j])
    
    msh_all = meshio.Mesh(vt, {'tetra': tt, 'triangle': face_all})
    msh_all.write('all.ply')