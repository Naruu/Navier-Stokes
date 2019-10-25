import numpy as np
import sympy as sym
import time
import sys
from tqdm import tqdm
import scipy.sparse.dok_matrix
import scipy.sparse.linalg


from fe1D_naive import basis, affine_mapping, u_glob
from numint import GaussLegendre

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
    1. compute A # compute on omega e only once. A does not change.
    2. for i=0, ...tn, # compute on omerga e and repeat for tn times
        1)compute b
        2)solve Ac = b
    """

    N_e = len(cells)
    N_n = np.array(dof_map).max() + 1

    A = scipy.sparse.dok_matrix((N_n, N_n))
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

            # Compute A_i,j(element matrix)
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
        #boundary condition
        A[0,:] = 0
        A[0,0] = 1
        

    for t in tqdm(range(nt)):
        b = np.zeros(N_n)
        for e in range(N_e):
            Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
            
            # Element vector
            b_e = np.zeros(n)
        
            for X, w in zip(points, weights):
              detJ = h/2
              dX = detJ*w
              x = affine_mapping(X, Omega_e)

              # Compute b_i(element vector)
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

    # boundary condition
        b[0] = c_n[-1]
        modified = True

        if verbose and modified:
            print('after essential boundary conditions:')
            print('b^(%d):' % e, b_e)
            
        timing['assemble'] = time.clock() - t0
        t1 = time.clock()
        c = scipy.sparse.linalg.spsolve(A.tocsr(), b, use_umfpack=True)
        cs.append(c)
        c_n = c
        timing['solve'] = time.clock() - t1
        if verbose:
            print('Global A:\n', A); print('Global b:\n', b)
            print('Solution c:\n', c)
    return cs, A, b, timing