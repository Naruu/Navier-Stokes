import time
import sys
import numpy as np
from math import exp
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
    lhs = []
    
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

        x1, lh = Lh(c_n, cells, vertices, dof_map, phi)
        time.sleep(0.1)
        x2, k1 = Lh(lh, cells, vertices, dof_map, phi)
        lhs.append(lh)
        time.sleep(0.1)
        K1 = [c_n[i] + dt * lh[i] for i in range(N_n)]
        c = [1/2 * c_n[i] + 1/2*(c_n[i] + dt* lh[i] + dt * k1[i]) for i in range(N_n)]
        time.sleep(0.1)
        """
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
    
        c = [c_n[r] + (K1[r]+2*K2[r]+2*K3[r]+K4[r])*dt/6 for r in range(len(c_n))]
        """
        # periodic boundary condition
        c[0] = c[-1]

        # set c for next time step
        c_n = c
        cs.append(c_n)

        #print("c", file = log)
        #print(c, file = log)    
        
    return cs, A, b, lhs

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
    L = 2  # total range: [0, L]
    d = 1  # d : order of polynomial
    N_e = 60  # N_e : number of elements
    dx = L / N_e  # spatial interval of an element
    nt = 20  # how many time points to compute?
    dt = 0.005  # time resolution


    # vetices: index of vertices
    # cells : a list of lists that contains vertex indexes in each cell
    # dof_map: cellwise index to global index mapping.
    # refer to p.110 of https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4screen.pdf
    vertices, cells, dof_map = mesh_uniform(
    N_e=N_e, d=d, Omega=[0,L])
    
    # Number of vertices in an element
    N_n = (np.array(dof_map).max() + 1)

    
    x0 = np.linspace(0,2,N_n)
    c0 = [exp((-10)*(x-1)**2) for x in x0]

    """
    # initial value of c
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
    """
    essbc = {}
    cs, A, b, lhs = finite_element1D_time(
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
        x,u, n_ = u_glob(cs[cc], cells, vertices, dof_map)
        plt.plot(x0, c0, 'ro')
        plt.plot(x, u, 'b-')
        plt.plot(x0, lhs[cc], 'g+')
        plt.show()


def Lh(c_n, cells, vertices, dof_map, phi):
    """
    Not yet implemented.
    It will compute Lh which is -u_x
    currently it returns u_h(1) of second order Runge-kutta
    """
    u_patches = [0] * len(c_n)
    x_patches = np.linspace(0,2,len(c_n))
    for e in range(len(cells)):
        Omega_e = (vertices[cells[e][0]], vertices[cells[e][-1]])
        d = len(dof_map[e]) - 1
        h = vertices[cells[e][-1]]-vertices[cells[e][0]]
        #X = np.linspace(-1, 1, resolution_per_element)
        X = np.linspace(-1,1,d+1)
        
        if d == 1:
            for r in range(d+1):
                i = dof_map[e][r] # global dof number
                u_patches[dof_map[e][0]:dof_map[e][0]+d+1] += [c_n[i]*phi[1][r](X,h), c_n[i]*phi[1][r](X,h)]
        else:
            for r in range(d+1):
                i = dof_map[e][r] # global dof number
                u_patches[dof_map[e][0]:dof_map[e][0]+d+1] += c_n[i]*phi[1][r](X,h)
        #print("i", i)
        u_patches[i] /= 2
    #print(u_patches)
    u_patches[-1] = 2 * u_patches[-1]
    #print(u_patches)
    u_patches = [(-1)* u for u in u_patches]
            #print("middle", r, c_n[i]*phi[1][r](X,h))
        #print("u_cell edge: {}, {}".format(u_cell[0],u_cell[-1]))
        #u_patches.append(u_cell)
    # Compute global coordinates of local nodes,
    # assuming all dofs corresponds to values at nodes
    return x_patches, u_patches

"""
x, u = Lh(c0, cells, vertices, dof_map, phi)
plt.plot(x, u)
print(x)
print(u)
plt.show()
c_n = c0
i = 0
r = 0
e = 0
"""

if __name__ == '__main__':
    main()