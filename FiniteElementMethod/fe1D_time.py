from fe1D_naive import GaussLegendre, basis, affine_mapping, mesh_uniform, u_glob
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import time
import sys
def finite_element1D_time(
    vertices, cells, dof_map,
    dt,
    nt,     # mesh
    essbc,                        # essbc[globdof]=value
    ilhs,
    irhs,
    initc = [0],
    blhs=lambda e, phi, r, s, X, x, h: 0,
    brhs=lambda e, phi, r, X, x, h: 0,
    intrule='GaussLegendre',
    verbose=False,
    ):

    """
    1. compute A # compute on omega e only once
    2. for i=0, ...tn, # compute on omerga e and repeat for tn times
        1)compute b
        2)solve Ac = b
    """

    N_e = len(cells)
    N_n = np.array(dof_map).max() + 1

    A = np.zeros((N_n, N_n))
    b = np.zeros(N_n)
    # Container to hold c
    cs = []

    # Polynomial degree
    # Compute all element basis functions and their derivatives
    
    h = vertices[cells[0][1]]-vertices[cells[0][0]]
    d = len(dof_map[0]) - 1
    phi = basis(d)
    n = d+1  # No of dofs per element
    
    # Integrate over the reference cell
    points, weights = GaussLegendre(d+1)
    timing = {}
    t0 = time.clock()
    
    """
    # initial value of c
    c_n = []
    X = np.linspace(-1, 1)
    for e in range(N_e):
        x = affine_mapping(X, Omega_e)
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        c_n.append(initf(x))
    """

    c_n = initc
    cs.append(c_n)

    for e in range(N_e):
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        
        # Element matrix
        A_e = np.zeros((n, n))
        
        for X, w in zip(points, weights):
            detJ = h/2
            dX = detJ*w
            x = affine_mapping(X, Omega_e)

            # Compute contribution to element matrix and vector
            for r in range(n):
                for s in range(n):
                    A_e[r,s] += ilhs(e, phi, r, s, X, x, h)*dX
        if verbose:
            print("original")
            print('A^(%d):\n' % e, A_e)

        # Assemble
        for r in range(n):
            for s in range(n):
                A[dof_map[e][r], dof_map[e][s]] += A_e[r,s]
        
        #boundary condidtion
        A[0,:] = 0
        A[:,0] = 0
        A[0,0] = 1


    for t in range(nt):
        for e in range(N_e):
            Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
            
            # Element vector
            b_e = np.zeros(n)
        
            for X, w in zip(points, weights):
              detJ = h/2
              dX = detJ*w
              x = affine_mapping(X, Omega_e)

              # Compute contribution to element matrix and vector
              for r in range(n):
                for s in range(n):
                    cc = c_n[dof_map[e][s]]
                    b_e[r] += irhs(e, phi, cc, r, s, X, x, h, dt)*dX
                    
            if verbose:
                print("original")
                print('b^(%d):' % e, b_e)

            # Assemble
            for r in range(n):
                b[dof_map[e][r]] += b_e[r]

    # Incorporate essential boundary conditions
        b[0] = c_n[-1]
        modified = True

        if verbose and modified:
            print('after essential boundary conditions:')
            print('b^(%d):' % e, b_e)
            
        timing['assemble'] = time.clock() - t0
        t1 = time.clock()
        c = np.linalg.solve(A, b)
        cs.append(c)
        c_n = c
        timing['solve'] = time.clock() - t1
        if verbose:
            print('Global A:\n', A); print('Global b:\n', b)
            print('Solution c:\n', c)
    return cs, A, b, timing

def ilhs(e, phi, r, s, X, x, h):
  return phi[0][r](X)*phi[0][s](X)

def irhs(e, phi, c, r, s, X, x, h, dt):
  return c*phi[0][r](X)*phi[0][s](X) - c*dt*phi[1][s](X, h)*phi[0][r](X)


def blhs(e, phi, r, s, X, x, h):
  return 0
def brhs(e, phi, r, X, x, h):
  return 0

"""
HOW TO

Define ilhs, rhs, blhs, brhs as following
blhs, brhs implies the boundary condition on first derviatives
essbc implies the boundary condition on u
"""

C = 5; D = 2; L = 1
d = 1; N_e = 10; dx = L/N_e
dt = 0.05; nt = int(1/dt)
 

vertices, cells, dof_map = mesh_uniform(
N_e=N_e, d=d, Omega=[0,L])

c0 = [0] * len(vertices)
i4 = int(0.4/dx)
i6 = int(0.6/dx)
c0[i4:i6+2] = [1] * (i6 - i4+2)

essbc = {}
#essbc[0] = c0[-1]

 
cs, A, b, timing = finite_element1D_time(
    vertices, cells, dof_map, dt, nt, essbc,
    ilhs=ilhs, irhs=irhs, initc=c0, blhs=blhs, brhs=brhs, verbose = False)

"""
For plotting

import matplotlib.pyplot as plt
x,u, nodes =  u_glob(c, cells, vertices, dof_map)
plt.plot(x, u)

exaxt solution
change u_exact as the model
u_exact = lambda x: D + C*(x-L) + (1./6)*(L**3 - x**3)
u_e = u_exact(nodes)
plt.plot(np.linspace(0,1,len(u_e)), u_e)


for cc in range(len(cs)):
    if cc%4 == 0:
        plt.figure()
        x,u, n_ = u_glob(cs[cc], cells, vertices, dof_map)
        plt.plot(x, u)
        plt.xticks(x)
        plt.yticks(u)
        plt.show()
"""

