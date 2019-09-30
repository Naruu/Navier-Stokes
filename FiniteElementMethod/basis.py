import sympy as sym
import numpy as np
from matplotlib import pyplot as plt
# least_square error  calculation betwen f and sigma from i to N c_i psi_i. {psi_i} composes basis.
# dot product is defined as integral.
# f is a target function to apporximate
# Omega is a tuple with two ends of integral domain
# psi is a list of basis elements.
def least_square(f, psi, Omega, symbolic=True):
    N = len(psi)
    A = sym.zeros(N, N)
    b = sym.zeros(N, 1)
    x = sym.Symbol('x')
    I = sym.integrate(0,(x, Omega[0], Omega[1]))
    if symbolic:
        for i in range(len(psi)):
            for j in range(i,len(psi)):
                I = sym.integrate(psi[i]*psi[j], (x, Omega[0], Omega[1]))
                A[i,j] = A[j,i] = I
            b[i] = sym.integrate(psi[i]*f, (x, Omega[0], Omega[1]))
        
        # Numerical solution
    if not symbolic or not isinstance(I, sym.Integral):
        for i in range(len(psi)):
            for j in range(i,len(psi)):
                integrand = psi[i] * psi[j]
                integrand = sym.lambdify([x], integrand)
                I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
                A[i,j] = A[j,i] = I
        integrand = psi[i] * f
        integrand = sym.lambdify([x], integrand)
        I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
        b[i,0] = I

    # Get coefficients of approximated function by solving system of equations
    c = A.LUSolve(b)
    c = [sym.simplify(c[i][0] for i in range(c.shape[0]))]
    # Construct the approximated function
    u = sum([c[i,0] * psi[i] for i in range(len(psi))])
    return u,c

def comparison_plot(f, u, Omega):
    x = sym.Symbol('x')
    f = sym.lambdify([x], f, modules='numpy')
    u = sym.lambdify([x], u, modules='numpy')
    resolution = 401
    xcoor = np.linspace(Omega[0], Omega[1], resolution)
    exact = f(xcoor)
    approx = u(xcoor)
    plt.plot(xcoor, approx)
    plt.hold('on')
    plt.plot(xcoor, exact)
    plt.legend(['approximation', 'exact'])


if __name__ == '__main__':
    x = sym.Symbol('x')
    f = 10*(x-1)**2 - 1
    u, c = least_square(f=f, psi=[1,x], Omega=[1,2])
    print(u)