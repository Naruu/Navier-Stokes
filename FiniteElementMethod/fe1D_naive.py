import numpy as np
import sympy as sym
import sys
import time

from numint import GaussLegendre

def finite_element1D_naive(
    vertices, cells, dof_map,     # mesh
    essbc,                        # essbc[globdof]=value
    ilhs,
    irhs,
    blhs=lambda e, phi, r, s, X, x, h: 0,
    brhs=lambda e, phi, r, X, x, h: 0,
    intrule='GaussLegendre',
    verbose=False,
    ):
    N_e = len(cells)
    N_n = np.array(dof_map).max() + 1

    A = np.zeros((N_n, N_n))
    b = np.zeros(N_n)

    timing = {}
    t0 = time.clock()

    for e in range(N_e):
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        h = Omega_e[1] - Omega_e[0]

        d = len(dof_map[e]) - 1  # Polynomial degree
        # Compute all element basis functions and their derivatives
        phi = basis(d)

        if verbose:
            print('e=%2d: [%g,%g] h=%g d=%d' % \
                  (e, Omega_e[0], Omega_e[1], h, d))

        # Element matrix and vector
        n = d+1  # No of dofs per element
        A_e = np.zeros((n, n))
        b_e = np.zeros(n)

        # Integrate over the reference cell
        if intrule == 'GaussLegendre':
            points, weights = GaussLegendre(d+1)
        elif intrule == 'NewtonCotes':
            points, weights = NewtonCotes(d+1)

        for X, w in zip(points, weights):
            detJ = h/2
            x = affine_mapping(X, Omega_e)
            dX = detJ*w

            # Compute contribution to element matrix and vector
            for r in range(n):
                for s in range(n):
                    A_e[r,s] += ilhs(e, phi, r, s, X, x, h)*dX
                b_e[r] += irhs(e, phi, r, X, x, h)*dX

        # Add boundary terms
        for r in range(n):
            for s in range(n):
                A_e[r,s] += blhs(e, phi, r, s, X, x, h)
            b_e[r] += brhs(e, phi, r, X, x, h)

        if verbose:
            print('A^(%d):\n' % e, A_e);  print('b^(%d):' % e, b_e)

        # Incorporate essential boundary conditions
        modified = False
        for r in range(n):
            global_dof = dof_map[e][r]
            if global_dof in essbc:
                # dof r is subject to an essential condition
                value = essbc[global_dof]
                # Symmetric modification
                b_e -= value*A_e[:,r]
                A_e[r,:] = 0
                A_e[:,r] = 0
                A_e[r,r] = 1
                b_e[r] = value
                modified = True

        if verbose and modified:
            print('after essential boundary conditions:')
            print('A^(%d):\n' % e, A_e);  print('b^(%d):' % e, b_e)

        # Assemble
        for r in range(n):
            for s in range(n):
                A[dof_map[e][r], dof_map[e][s]] += A_e[r,s]
            b[dof_map[e][r]] += b_e[r]

    timing['assemble'] = time.clock() - t0
    t1 = time.clock()
    c = np.linalg.solve(A, b)
    timing['solve'] = time.clock() - t1
    if verbose:
        print('Global A:\n', A); print('Global b:\n', b)
        print('Solution c:\n', c)
    return c, A, b, timing



def affine_mapping(X, Omega_e):
    x_L, x_R = Omega_e
    return 0.5*(x_L + x_R) + 0.5*(x_R - x_L)*X

def Lagrange_polynomial(x, i, points):
    """
    Return the Lagrange polynomial no. i.
    points are the interpolation points, and x can be a number or
    a sympy.Symbol object (for symbolic representation of the
    polynomial). When x is a sympy.Symbol object, it is
    normally desirable (for nice output of polynomial expressions)
    to let points consist of integers or rational numbers in sympy.
    """
    p = 1
    for k in range(len(points)):
        if k != i:
            p *= (x - points[k])/(points[i] - points[k])
    return p

    return nodes

def basis(d, symbolic=False):
    """
    Return all local basis function phi and their derivatives,
    in physical coordinates, as functions of the local point
    X in a 1D element with d+1 nodes.
    If symbolic=True, return symbolic expressions, else
    return Python functions of X.
    point_distribution can be 'uniform' or 'Chebyshev'.
    >>> phi = basis(d=1, symbolic=False)
    >>> phi[0][0](0)  # basis func 0 at X=0
    0.5
    >>> phi[1][0](0, h=0.5)  # 1st x-derivative at X=0
    -2
    """
    X, h = sym.symbols('X h')
    phi_sym = {}
    phi_num = {}
    if d == 0:
        phi_sym[0] = [1]
        phi_sym[1] = [0]
    else:
        nodes = np.linspace(-1, 1, d+1)

        phi_sym[0] = [Lagrange_polynomial(X, r, nodes)
                      for r in range(d+1)]
        phi_sym[1] = [sym.simplify(sym.diff(phi_sym[0][r], X)*2/h)
                      for r in range(d+1)]
    # Transform to Python functions
    phi_num[0] = [sym.lambdify([X], phi_sym[0][r])
                  for r in range(d+1)]
    phi_num[1] = [sym.lambdify([X, h], phi_sym[1][r])
                  for r in range(d+1)]
    return phi_sym if symbolic else phi_num

def mesh_uniform(N_e, d, Omega):
    """
    Create 1D finite element mesh on Omega with N_e elements
    of the polynomial degree d.

    Input
    N_e : numer of elements
    d : degree of basis polynomial
    omega : domain

    Return
    vertices: verticies
    cells: local vertex to global vertex  mapping
    dof_map: local to global degree of freedom mapping
    """
    vertices = np.linspace(Omega[0], Omega[1], N_e + 1).tolist()
    if d == 0:
        dof_map = [[e] for e in range(N_e)]
    else:
        dof_map = [[e*d + i for i in range(d+1)] for e in range(N_e)]
    cells = [[e, e+1] for e in range(N_e)]
    return vertices, cells, dof_map

def u_glob(U, cells, vertices, dof_map, resolution_per_element=1):
    """
    Compute (x, y) coordinates of a curve y = u(x), where u is a
    finite element function: u(x) = sum_i of U_i*phi_i(x).
    (The solution of the linear system is in U.)
    Method: Run through each element and compute curve coordinates
    over the element.
    This function works with cells, vertices, and dof_map.
    """
    x_patches = []
    u_patches = []
    nodes = {}  # node coordinates (use dict to avoid multiple values)
    for e in range(len(cells)):
        Omega_e = (vertices[cells[e][0]], vertices[cells[e][-1]])
        d = len(dof_map[e]) - 1
        phi = basis(d)
        X = np.linspace(-1, 1, resolution_per_element)
        x = affine_mapping(X, Omega_e)
        x_patches.append(x)
        u_cell = 0  # u(x) over this cell
        for r in range(d+1):
            i = dof_map[e][r]  # global dof number
            u_cell += U[i]*phi[0][r](X)
        u_patches.append(u_cell)
        # Compute global coordinates of local nodes,
        # assuming all dofs corresponds to values at nodes
        X = np.linspace(-1, 1, d+1)
        x = affine_mapping(X, Omega_e)
        for r in range(d+1):
            nodes[dof_map[e][r]] = x[r]
    nodes = np.array([nodes[i] for i in sorted(nodes)])
    x = np.concatenate(x_patches)
    u = np.concatenate(u_patches)
    return x, u, nodes


"""
HOW TO

Define ilhs, rhs, blhs, brhs as following
blhs, brhs implies the boundary condition on first derviatives
essbc implies the boundary condition on u

C = 5; D = 2; L = 2
d = 1

def ilhs(e, phi, r, s, X, x, h):
  return phi[1][r](X, h)*phi[1][s](X, h)
def irhs(e, phi, r, X, x, h):
  return x**2*phi[0][r](X)
def blhs(e, phi, r, s, X, x, h):
  return 0
def brhs(e, phi, r, X, x, h):
  return -C*phi[0][r](-1) if e == 0 else 0


vertices, cells, dof_map = mesh_uniform(
N_e=10, d=d, Omega=[0,L])


essbc = {}
essbc[dof_map[-1][-1]] = D
c, A, b, timing = finite_element1D_naive(
    vertices, cells, dof_map, essbc,
    ilhs=ilhs, irhs=irhs, blhs=blhs, brhs=brhs)
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