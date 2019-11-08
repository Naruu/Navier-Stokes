import time
import sys
import numpy as np
import sympy as sym
#from scipy.sparse import dok_matrix, linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

from fe1D_naive import affine_mapping, basis, mesh_uniform, u_glob
from numint import GaussLegendre

global log
# print out result to file
log = open('print_log_euler.txt', 'w')

# initialzing c with
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
    nt, 
    essbc,
    ilhs,
    irhs,
    c0 = [0],
    blhs=lambda e, phi, r, s, X, x, h: 0,
    brhs=lambda e, phi, r, X, x, h: 0,
    intrule='GaussLegendre',
    verbose=False,
    ):

    """
    Refer to p.245 of 
    https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4screen.pdf
    1. compute A # compute on omega e only once. A does not change.
    2. for i=0, ...tn, # compute on omerga e and repeat for tn times
        1)compute b
        2)solve Ac = b
    Use Euler method for time discretization
    """
    # Numner of elements
    N_e = len(cells)
    # Number of vertices in an element
    N_n = np.array(dof_map).max() + 1

    # matrix for global computation
    A = np.zeros((N_n, N_n))
    # A = dok_matrix((N_n, N_n)) # sparse matrix
    b = np.zeros(N_n)

    # Container to hold c
    cs = []
    
    # initializing c
    c_n = c0
    #c_n= init_c(A, N_n, c0)
    cs.append(c_n)

    # Polynomial degree
    # Compute all element basis functions and their derivatives
    d = len(dof_map[0]) - 1
    phi = basis(d)
    n = d+1  # No of dofs per element


    # Integration method : GaussLegendre
    # points: x coordinate, weights: weight at each point
    points, weights = GaussLegendre(d+1)

    # Integrate over the reference cell
    for e in range(N_e):
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        h = Omega_e[1] - Omega_e[0]
    
        # matrix for elementwise computation
        # Element matrix
        A_e = np.zeros((n, n))

        for X, w in zip(points, weights):
            detJ = h/2
            dX = detJ*w
            x = affine_mapping(X, Omega_e)
        
            # Compute A_i,j(element matrix), B_i,j
            for r in range(n):
                for s in range(n):
                    A_e[r,s] += ilhs(e, phi, r, s, X, x, h)*dX

        # Assemble
        # Map elementwise matrix to global matrix
        for r in range(n):
            for s in range(n):
                A[dof_map[e][r], dof_map[e][s]] += A_e[r,s]
        
        # periodic boundary condition
        A[-1,:] = 0
        A[-1,-1] = -1
        A[-1, 0] = 1
    

    for t in tqdm(range(nt)):
        # matrix for global computation
        b = np.zeros(N_n)
        for e in range(N_e):
            Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
            h = Omega_e[1] - Omega_e[0]
            
            # matrix for elementwise computation
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
                        # c_n value that match to the vertice
                        cc = c_n[dof_map[e][s]]
                        b_e[r] += irhs(e, phi, cc, r, s, X, x, h, dt)*dX
                        #print("r: {}, s: {}, dof_r : {}, dof_s : {}".format(r,s,dof_map[e][r],dof_map[e][s]), file = log)
                        #print("phi[0][r](X): {}, phi[0][s](X): {}".format(phi[0][r](X), phi[0][s](X)), file=log)

            # Assemble
            for r in range(n):
                b[dof_map[e][r]] += b_e[r]

            print("b", file = log)
            print(b, file = log)

        # boundary condition
        b[0] = 0

        c = np.linalg.solve(A, b)
        # sparse matrix version
        #c = linalg.spsolve(A.tocsr(), b, use_umfpack=True)
        
        """
        # record for every 10th timestep
        if not t%10:
            cs.append(c)
        """
        cs.append(c)
        c_n = c

    return cs, A, b


#left hand side : matix A_ij
def ilhs(e, phi, r, s, X, x, h):
    return phi[0][r](X)*phi[0][s](X)

def irhs(e, phi, c, r, s, X, x, h, dt):
    return c*phi[0][r](X)*phi[0][s](X) - dt*c*phi[1][s](X, h)*phi[0][r](X)

def blhs(e, phi, r, s, X, x, h):
    return 0

def brhs(e, phi, r, X, x, h):
    return 0
    
def main():
    plt.clf()
    plt.close('all')
    
    # total range: [0, L]
    # d : order of polynomial
    # N_e : number of elements

    L = 1; d = 1
    N_e = 20; dx = L/N_e
    nt = 1; dt = 0.001

    # vetices: index of vertices
    # cells : a list of lists that contains vertex indexes in each cell
    # dof_map: cellwise index to global index mapping.
    # refer to p.110 of https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4screen.pdf
    vertices, cells, dof_map = mesh_uniform(
    N_e=N_e, d=d, Omega=[0,L])

    # Number of vertices in an element
    N_n = (np.array(dof_map).max() + 1)
    
    # initial value of c
    c0 = [0] * N_n 
    i4 = int(0.4 * N_n)
    i6 = int(0.6 * N_n)
    c0[i4:i6+1] = [1] * (i6 - i4+1)

    essbc = {}
    
    cs, A, b = finite_element1D_time(
        vertices, cells, dof_map, dt, nt, essbc,
        ilhs=ilhs, irhs=irhs, c0=c0, blhs=blhs, brhs=brhs, verbose=False)

    #Plot
    print("order of Legendre Polynomial: {}".format(d))
    print("N_e: {}, dx: {}, dt: {}".format(N_e, dx, dt))
    xtp = [L/6*x for x in range(7)]
    xlabel = ["{:.1}".format(L/6*x) for x in range(7)]

    plt.figure()
    for cc in range(len(cs)):
        x,u, n_ = u_glob(cs[cc], cells, vertices, dof_map)
        plt.plot(x, u)
        plt.xlim(0,L)
        plt.xticks(xtp,xlabel)
        plt.show()

if __name__ == '__main__':
    main()