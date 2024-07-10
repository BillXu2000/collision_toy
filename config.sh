dim 3
dt 1e-2

# mesh ./scratch/tet.msh 0 0.5 0
mesh ./scratch/tet.msh 0 0.6 0
fixed 0: 
    point 0.2 0.59 0.2


$young 1e5

gravity {"gravity": [0, -9.8, 0]}
# floor {"young": $young}
neohookean {"young": $young, "nu": 0.3}
collision {"young": $young, "d_m": 1e-2}

initialize

# record ./scratch/test.log
load ./scratch/test.log
pause
frame 100

test_gradient
exit
