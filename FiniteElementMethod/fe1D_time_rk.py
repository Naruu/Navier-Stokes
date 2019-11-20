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
log = open('print_log_rk.txt', 'w')

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
    Refer to the formula at page 110 of 
    http://fab.cba.mit.edu/classes/864.14/text/fea.pdf
    Compute A, B.
    Use 4th order Runge-Kutta method for time discretizaiton
    """
    # Numner of elements
    N_e = len(cells)
    # Number of vertices in an element
    N_n = np.array(dof_map).max() + 1

    # matrix for global computation
    A = np.zeros((N_n, N_n))
    B = np.zeros((N_n, N_n))
    # A = dok_matrix((N_n, N_n)) # sparse matrix
    b = np.zeros(N_n)

    # Container to hold c
    cs = []
    
    # initializing c
    c_n = c0
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
        A_e = np.zeros((n, n)) # Element matrix
        B_e = np.zeros((n,n))

        for X, w in zip(points, weights):
            detJ = h/2
            dX = detJ*w
            x = affine_mapping(X, Omega_e)
        
            # Compute A_i,j(element matrix), B_i,j
            for r in range(n):
                for s in range(n):
                    A_e[r,s] += ilhs(e, phi, r, s, X, x, h)*dX
                    B_e[r,s] += irhs(e, phi, r, s, X, x, h)*dX

        # Assemble
        # Map elementwise matrix to global matrix
        for r in range(n):
            for s in range(n):
                A[dof_map[e][r], dof_map[e][s]] += A_e[r,s]
                B[dof_map[e][r], dof_map[e][s]] += B_e[r,s]

    # Check A,B values
    print("A", file = log)
    print(A, file = log)
    print("B", file = log)
    print(B, file = log)

    # c: coefficients. vector form.
    # dc/dt = M * c
    # check p.110 of http://fab.cba.mit.edu/classes/864.14/text/fea.pdf for theory
    A_inv = np.linalg.inv(A)
    M = np.matmul(A_inv,B)
    M = (-1)*M
    print("M", file = log)
    print(M, file = log)
    

    # Compute by 4th order Runge-Kutta
    for t in tqdm(range(nt)):
        K1 = K2 = K3 = K4 = np.zeros(N_n)
        K1 = np.matmul(M, c_n)
        K2 = np.matmul(M, c_n + K1*dt/2)
        K3 = np.matmul(M, c_n + K2*dt/2)
        K4 = np.matmul(M, c_n + K3*dt)
        
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
        # periodic boundary condition
        c[0] = c[-1]

        # set c for next time step
        c_n = c
        cs.append(c_n)

        print("c", file = log)
        print(c, file = log)    
        
    return cs, A, b

# left hand side : matix A_ij
def ilhs(e, phi, r, s, X, x, h):
    return phi[0][r](X)*phi[0][s](X)

def irhs(e, phi, r, s, X, x, h):
    return phi[0][r](X)* phi[1][s](X, h)

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