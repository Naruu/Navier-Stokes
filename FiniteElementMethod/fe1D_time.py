import time
import sys
import numpy as np
import sympy as sym
from math import exp, sin
#from scipy.sparse import dok_matrix, linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

from imp import reload
import numint, fe1D_naive
reload(numint)
reload(fe1D_naive)
from fe1D_naive import affine_mapping, basis, mesh_uniform, u_glob
from numint import GaussLegendre, NewtonCotes

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

    # Integration method
    # points: x coordinate, weights: weight at each point
    points, weights = GaussLegendre(d+1)
    # points, weights = NewtonCotes(d+1)

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
        #A[-1,:] = 0
        #A[-1,-1] = -1
        #A[-1, 0] = 1
        plt.figure('A matrix')
        plt.imshow(A, cmap='jet')  
        print(A, file=log)
    

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
            
                #print("X: {}, w: {}, x: {}".format(X, w, x), file=log)
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

            #print("b", file = log)
            #print(b, file = log)

        # boundary condition
        #b[0] = 0

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


    L = 2  # total range: [0, L]
    d = 2  # d : order of polynomial
    N_e = 40  # N_e : number of elements
    dx = L / N_e  # spatial interval of an element
    nt = 400  # how many time points to compute?
    dt = 0.0005  # time resolution

    # vetices: index of vertices
    # cells : a list of lists that contains vertex indexes in each cell
    # dof_map: cellwise index to global index mapping.
    # refer to p.110 of https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4screen.pdf
    vertices, cells, dof_map = mesh_uniform(
    N_e=N_e, d=d, Omega=[0,L])

    # Number of vertices in an element
    N_n = (np.array(dof_map).max() + 1)
    
    """
    x0 = np.linspace(0,2,N_n)
    c0 = [exp((-10)*(x-1)**2) for x in x0]
    """
    # trapezoid
    c0 = [0] * N_n
    x1 = int(0.3*N_n)
    x2 = int(0.4*N_n)
    x3 = int(0.5*N_n)
    x4 = int(0.6*N_n)

    x_i = np.arange(0, N_n, 1)

    c0[0:x1] = np.zeros(x1)
    x_ = (x_i - x1) / (x2 - x1)
    c0[x1:x2] = x_[x1:x2]
    c0[x2:x3] = np.ones((x3-x2))
    x_ = (x4 - x_i) / (x4 - x3)
    c0[x3:x4] = x_[x3:x4]
    c0[x4:] = np.zeros((N_n-x4))
    
    essbc = {}
    phi = basis(d)
    plt.figure('phi function')
    x2 = np.linspace(-1, 1, 20)
    color_list = ['b--', 'r-.', 'g-', 'y-']

    for jj in range(d+1):
        try:
            plt.plot(x2, phi[0][jj](x2), color_list[jj])
        except ValueError:
            print("phi[0][{}] is constant. phi[0][{}] = {}".format(jj,jj,phi[0][jj](1)))
    plt.show()
    plt.close()
    
    cs, A, b = finite_element1D_time(
        vertices, cells, dof_map, dt, nt, essbc,
        ilhs=ilhs, irhs=irhs, c0=c0, blhs=blhs, brhs=brhs, verbose=False)

    #Plot
    print("order of Legendre Polynomial: {}".format(d))
    print("N_e: {}, dx: {}, dt: {}".format(N_e, dx, dt))
    xtp = [L/6*x for x in range(7)]
    xlabel = ["{:.1}".format(L/6*x) for x in range(7)]

    x0 = np.linspace(0, L, N_n)
    plt.figure()
    for cc in range(0, len(cs), 20):
        #x,u, n_ = u_glob(cs[cc], cells, vertices, dof_map)
        plt.plot(x0, c0, 'ro')
        #plt.plot(x, u, 'b-')
        plt.plot(x0, cs[cc], 'g+')
        plt.show()
    return x, cs

if __name__ == '__main__':
    x, cs = main()
    log.close()