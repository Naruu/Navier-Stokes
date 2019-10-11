import numpy as np
import sympy as sym
import sys

def Lagrange_polynomials(x, i, points):
    p = 1
    for k in range(len(points)):
        if k != i:
            p *= (x - points[k])/(points[i] - points[k])
    return p

def basis(d, symbolic=True):
    """
    Return all local basis function phi as functions of the
    local point X in a 1D element with d+1 nodes.
    If symbolic=True, return symbolic expressions, else
    return Python functions of X.
    point_distribution can be 'uniform' or 'Chebyshev'.
    """
    X = sym.symbols('X')
    if d == 0:
      phi_sym = [1]
    else:
      if symbolic:
        h = sym.Rational(1, d)  # node spacing
        nodes = [2*i*h - 1 for i in range(d+1)]
      else:
        nodes = np.linspace(-1, 1, d+1)
        
      phi_sym = [Lagrange_polynomials(X, r, nodes) for r in range(d+1)]
      
     # Transform to Python functions
    phi_num = [sym.lambdify([X], phi_sym[r], modules='numpy') for r in range(d+1)]
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

def element_matrix(phi, Omega_e, symbolic=True, numint=None):
    n = int(len(phi))
    A_e = sym.zeros(n,n)
    X = sym.Symbol('X')
    if symbolic:
        h = sym.Symbol('h')
    else:
        h = Omega_e[1] - Omega_e[0]
    detJ = h/2  # dx/dX
    for r in range(n):
        for s in range(r, n):
            A_e[r,s] = sym.integrate(phi[r]*phi[s]*detJ, (X, -1, 1))
            A_e[s,r] = A_e[r,s]
    return A_e

def element_vector(f, phi, Omega_e, symbolic=True, numint=None):
    n = len(phi)
    b_e = sym.zeros(n, 1)
    # Make f a function of X (via f.subs to avoid real numbers from lambdify)
    X = sym.Symbol('X')
    if symbolic:
        h = sym.Symbol('h')
    else:
        h = Omega_e[1] - Omega_e[0]
    x = (Omega_e[0] + Omega_e[1])/2 + h/2*X  # mapping
    f = f.subs('x', x)
    detJ = h/2
    for r in range(n):
        if symbolic:
            I = sym.integrate(f*phi[r]*detJ, (X, -1, 1))
        if not symbolic or isinstance(I, sym.Integral):
            # Ensure h is numerical
            h = Omega_e[1] - Omega_e[0]
            detJ = h/2
            f_func = sym.lambdify([X], f, 'mpmath')
            # phi is function
            integrand = lambda X: f_func(X)*phi[r](X)*detJ
            #integrand = integrand.subs(sym.pi, np.pi)
            # integrand may still contain symbols like sym.pi that
            # prevents numerical evaluation...
            try:
                I = mpmath.quad(integrand, [-1, 1])
            except Exception as e:
                print('Could not integrate f*phi[r] numerically:')
                print(e)
                sys.exit(0)
        b_e[r] = I
    return b_e

def assemble(vertices, cells, dof_map, phi, f,
             symbolic=True, numint=None):
    N_n = len(list(set(np.array(dof_map).ravel())))
    N_e = len(cells)
    if symbolic:
        A = sym.zeros(N_n, N_n)
        b = sym.zeros(N_n, 1)    # note: (N_n, 1) matrix
    else:
        A = np.zeros((N_n, N_n))
        b = np.zeros(N_n)
    for e in range(N_e):
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        A_e = element_matrix(phi[e], Omega_e, symbolic, numint)
        b_e = element_vector(f, phi[e], Omega_e, symbolic, numint)
        #print 'element', e
        #print b_e
        for r in range(len(dof_map[e])):
            for s in range(len(dof_map[e])):
                A[dof_map[e][r],dof_map[e][s]] += A_e[r,s]
            b[dof_map[e][r]] += b_e[r]
    return A, b

    def approximate(f, symbolic=False, d=1, N_e=4, numint=None,
                Omega=[0, 1], collocation=False, filename='tmp'):
    """
    Compute the finite element approximation, using Lagrange
    elements of degree d, to a symbolic expression f (with x
    as independent variable) on a domain Omega. N_e is the
    number of elements.
    symbolic=True implies symbolic expressions in the
    calculations, while symbolic=False means numerical
    computing.
    numint is the name of the numerical integration rule
    (Trapezoidal, Simpson, GaussLegendre2, GaussLegendre3,
    GaussLegendre4, etc.). numint=None implies exact
    integration.
    """
    numint_name = numint  # save name
    if symbolic:
      numint = [[sym.S(-1), sym.S(1)], [sym.S(1), sym.S(1)]]  # sympy integers
    else:
      numint = [[-1, 1], [1, 1]]

    vertices, cells, dof_map = mesh_uniform(N_e, d, Omega)

    # phi is a list where phi[e] holds the basis in cell no e
    # (this is required by assemble, which can work with
    # meshes with different types of elements).
    # len(dof_map[e]) is the number of nodes in cell e,
    # and the degree of the polynomial is len(dof_map[e])-1
    phi = [basis(len(dof_map[e])-1) for e in range(N_e)]
    numint = None
    A, b = assemble(vertices, cells, dof_map, phi, f, symbolic=symbolic, numint=numint)

    print('cells:', cells)
    print('vertices:', vertices)
    print('dof_map:', dof_map)
    print('A:\n', A)
    print('b:\n', b)
    #print sym.latex(A, mode='plain')
    #print sym.latex(b, mode='plain')

    if symbolic:
        c = A.LUsolve(b)
        c = np.asarray([c[i,0] for i in range(c.shape[0])])
    else:
        c = np.linalg.solve(A, b)

    print('c:\n', c)
    """

    x = sym.Symbol('x')
    f = sym.lambdify([x], f, modules='numpy')

    title = 'P%d, N_e=%d' % (d, N_e)
    x_u, u, _ = u_glob(c, vertices, cells, dof_map,
                       resolution_per_element=51)
    x_f = np.linspace(Omega[0], Omega[1], 10001) # mesh for f
    import scitools.std as plt
    plt.plot(x_u, u, '-',
             x_f, f(x_f), '--')
    plt.legend(['u', 'f'])
    plt.title(title)
    """
    return c