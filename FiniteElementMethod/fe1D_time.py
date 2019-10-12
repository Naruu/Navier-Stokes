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
            print('A^(%d):\n' % e, A_e);  print('b^(%d):' % e, b_e)

        # Assemble
        for r in range(n):
            for s in range(n):
                A[dof_map[e][r], dof_map[e][s]] += A_e[r,s]
        
        #boundary condidtion
        A[0,:] = 0
        A[:,0] = 0
        A[0,0] = 1 * N_e


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
                    b_e[r] += irhs(e, phi, c_n, r, s, X, x, h, dt)*dX
                    
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
        print('A^(%d):\n' % e, A_e);  print('b^(%d):' % e, b_e)
        
    timing['assemble'] = time.clock() - t0
    t1 = time.clock()
    c = np.linalg.solve(A, b)
    c_n = c
    timing['solve'] = time.clock() - t1
    if verbose:
        print('Global A:\n', A); print('Global b:\n', b)
        print('Solution c:\n', c)
    return c, A, b, timing

def ilhs(e, phi, r, s, X, x, h):
  return phi[1][r](X, h)*phi[1][s](X, h)

def irhs(e, phi, c_n, r, s, X, x, h, dt):
  print(phi[0][r](X,h))
  return c_n[r]*phi[0][r](X, h)*phi[0][s](X, h) + c_n[r]*dt*phi[1][r](X, h)*phi[0][s](X, h)

"""
HOW TO

Define ilhs, rhs, blhs, brhs as following
blhs, brhs implies the boundary condition on first derviatives
essbc implies the boundary condition on u

C = 5; D = 2; L = 2
d = 1
dt = 1; nt = 10


vertices, cells, dof_map = mesh_uniform(
N_e=10, d=d, Omega=[0,L])

c0 = [0] * len(vertices)
c0[4] = c0[5] = c0[6] = 1

essbc = {}
essbc[0] = c0[-1]

c, A, b, timing = finite_element1D_time(
    vertices, cells, dof_map, dt, nt, essbc,
    ilhs=ilhs, irhs=irhs, initc=c0, blhs=blhs, brhs=brhs)
print(A)
print(b)
print(c)
"""

"""
For plotting

import matplotlib.pyplot as plt
x,u, nodes = u_glob(c, cells, vertices, dof_map)
plt.plot(x, u)

exaxt solution
change u_exact as the model
u_exact = lambda x: D + C*(x-L) + (1./6)*(L**3 - x**3)
u_e = u_exact(nodes)
plt.plot(np.linspace(0,1,len(u_e)), u_e)
"""
