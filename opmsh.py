import meshio, sys
import numpy as np

fn = sys.argv[1]

with open(fn, 'r') as fi:
    lines = fi.readlines()

line_sf = []
flag_sf = False

for line in lines:
    if flag_sf:
        line_sf.append(line)
    if 'Surface' in line:
        flag_sf = not flag_sf

if 'Surface' in line_sf[-1]:
    line_sf = line_sf[:-1]
line_sf = line_sf[1:]
line_sf = [[int(x) for x in line.split()] for line in line_sf]
line_sf = np.array(line_sf)

old = meshio.read(fn)
ms = meshio.Mesh(old.points, {'triangle': line_sf - 1, 'tetra': old.cells[0].data})
ms.write('new.msh', binary=False)
