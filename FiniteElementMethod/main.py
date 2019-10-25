import matplotlib.pyplot as plt
import numpy as np

from fe1D_naive import mesh_uniform,u_glob
from fe1D_time import finite_element1D_time
"""
HOW TO

Define ilhs, rhs, blhs, brhs as following
blhs, brhs implies the boundary condition on first derviatives
essbc implies the boundary condition on u
"""

def ilhs(e, phi, r, s, X, x, h):
  return phi[0][r](X)*phi[0][s](X)

def irhs(e, phi, c, r, s, X, x, h, dt):
  return c*phi[0][r](X)*phi[0][s](X) - dt*c*phi[1][s](X, h)*phi[0][r](X)

def blhs(e, phi, r, s, X, x, h):
  return 0
def brhs(e, phi, r, X, x, h):
  return 0

L = 1; d = 4
N_e = 50; dx = L/N_e
dt = 0.00001; nt = 10000
 

vertices, cells, dof_map = mesh_uniform(
N_e=N_e, d=d, Omega=[0,L])

N_n = (np.array(dof_map).max() + 1)

c0 = [0] * N_n 
i4 = int(0.4 * N_n)
i6 = int(0.6 * N_n )
c0[i4:i6+2] = [1] * (i6 - i4+2)

essbc = {}
#essbc[0] = c0[-1]
 
cs, A, b, timing = finite_element1D_time(
    vertices, cells, dof_map, dt, nt, essbc,
    ilhs=ilhs, irhs=irhs, initc=c0, blhs=blhs, brhs=brhs, verbose = False)

#Plot
xtp = [L/6*x for x in range(7)]
xlabel = ["{:.1}".format(L/6*x) for x in range(7)]

for cc in range(5):
    plt.figure()
    x,u, n_ = u_glob(cs[nt//5 *cc], cells, vertices, dof_map)
    plt.plot(x, u)
    plt.xlim(0,L)
    plt.xticks(xtp,xlabel)
    #plt.yticks(u)
    plt.show()

"""
For plotting

import matplotlib.pyplot as plt
x,u, nodes =  u_glob(c, cells, vertices, dof_map)
plt.plot(x, u)

exaxt solution
change u_exact as the model
u_exact = lambda x: D + C*(x-L) + (1./6)*(L**3 - x**3)
u_e = u_exact(nodes)
plt.plot(np.linspace(0,1,len(u_e)), u_e)


xtp = [L/10*x for x in range(11)]
xlabel = [str(L/10*x) for x in range(11)]

for cc in range(0,len(cs),20):
    plt.figure()
    x,u, n_ = u_glob(cs[cc], cells, vertices, dof_map)
    plt.plot(x, u)
    plt.xlim(0,L)
    plt.xticks(xtp,xlabel)
    #plt.yticks(u)
    plt.show()
"""
