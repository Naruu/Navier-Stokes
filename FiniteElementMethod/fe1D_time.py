from math import cos, sin
import time
import sys
import numpy as np
import sympy as sym
#from scipy.sparse import dok_matrix, linalg
from tqdm import tqdm

from fe1D_naive import basis, affine_mapping, u_glob
from numint import GaussLegendre

global log
log = open('print_log.txt', 'w')

def init_c(A, N_n, c0):
    c_n = np.zeros(N_n)
    for r in range(N_n):
        for s in range(N_n):
            c_n[r] += A[r,s]*c0[s]
    print("initial c", file = log)
    print(c_n, file = log)
    return c_n


def finite_element1D_time(
    vertices, cells, dof_map,
    dt,
    nt,     # mesh
    essbc,                        # essbc[globdof]=value
    ilhs,
    irhs1,
    irhs2,
    c0 = [0],
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

    A = np.zeros((N_n, N_n))
    B = np.zeros((N_n, N_n))
    #A = dok_matrix((N_n, N_n))
    b = np.zeros(N_n)
    # Container to hold c
    cs = []
    
    # Polynomial degree
    # Compute all element basis functions and their derivatives
    
    #h = vertices[cells[0][1]]-vertices[cells[0][0]]
    d = len(dof_map[0]) - 1
    phi = basis(d)
    n = d+1  # No of dofs per element
    h = vertices[cells[0][1]] - vertices[cells[0][0]]
    
    # Integrate over the reference cell
    points, weights = GaussLegendre(d+1)

    """
    # initial value of c
    c_n = []
    X = np.linspace(-1, 1)
    for e in range(N_e):
        x = affine_mapping(X, Omega_e)
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        c_n.append(initf(x))
    """

    for e in range(N_e):
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]

        # Element matrix
        A_e = np.zeros((n, n))
        B_e = np.zeros((n,n))
        
        for X, w in zip(points, weights):
            detJ = h/2
            dX = detJ*w
            x = affine_mapping(X, Omega_e)

            # Compute A_i,j(element matrix)
            for r in range(n):
                for s in range(n):
                    A_e[r,s] += ilhs(e, phi, r, s, X, x, h)*dX
                    B_e[r,s] += irhs2(e, phi, r, s, X, x, h)*dX
        """
        if verbose:
            print("original")
            print('A^(%d):\n' % e, A_e)
        """


        # Assemble
        for r in range(n):
            for s in range(n):
                A[dof_map[e][r], dof_map[e][s]] += A_e[r,s]
                B[dof_map[e][r], dof_map[e][s]] += B_e[r,s]
                
    print("A", file = log)
    print(A, file = log)
    print("B", file = log)
    print(B, file = log)
    
    #print("c0: ",c0)
    #c_n = init_c(A, N_n, c0)
    #print("c_n",c_n)
    c_n = c0
    cs.append(c_n)

    
    A_inv = np.linalg.inv(A)
    M = np.matmul(A_inv,B)
    M = -M
    print("M", file = log)
    print(M, file = log)
    

    for t in tqdm(range(nt)):
        K1 = K2 = K3 = K4 = np.zeros(N_n)
        K1 = np.matmul(M, c_n)
        K2 = np.matmul(M, c_n + K1*h/2)
        K3 = np.matmul(M, c_n + K2*h/2)
        K4 = np.matmul(M, c_n + K3*h)
        
        print("K1", file = log)
        print(K1, file = log)
        print("K2", file = log)
        print(K2, file = log)
        print("K2", file = log)
        print(K2, file = log)
        print("K3", file = log)
        print(K3, file = log)
        print("K4", file = log)
        print(K4, file = log)
    
        c = [c_n[r] + (K1[r]+2*K2[r]+2*K3[r]+K4[r])*h/6 for r in range(len(c_n))]
        c[-1] = c[0]
        c_n = c
        cs.append(c_n)
        print("c", file = log)
        print(c, file = log)    

    """            
        c = np.linalg.solve(A, b)
        cs.append(c)
        c_n = c
    
    for t in tqdm(range(nt)):
        print(t, file = log)
        b = np.zeros(N_n)
        for e in range(N_e):
            Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
            # Element vector
            b_e = np.zeros(n)
        
            for X, w in zip(points, weights):
                detJ = h/2
                dX = detJ*w
                x = affine_mapping(X, Omega_e)
            
                print("X: {}, w: {}, x: {}".format(X, w, x), file=log)
                # Compute b_i(element vector)
                for r in range(n):
                    for s in range(n):
                        print("r: {}, s: {}, dof_r : {}, dof_s : {}".format(r,s,dof_map[e][r],dof_map[e][s]), file = log)
                        cc = c_n[dof_map[e][s]]
                        print("phi[0][r](X): {}, phi[0][s](X): {}".format(phi[0][r](X), phi[0][s](X)), file=log)
                        b_e[r] += irhs(e, phi, cc, r, s, X, x, h, dt)*dX    

            # Assemble
            for r in range(n):
                b[dof_map[e][r]] += b_e[r]
            print("b_e", file = log)
            print(b_e, file = log)
            print("b", file = log)
            print(b, file = log)

        # boundary condition
        #b[0] = 0
        #modified = True

        c = np.linalg.solve(A, b)
        #c = linalg.spsolve(A.tocsr(), b, use_umfpack=True)
        #if t < 5:
        cs.append(c)
        c_n = c
        
    """
    return cs, A, b