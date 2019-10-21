- numint.py -> GaussLegendre polynomial for basis function
- fe1D_naive.py -> supporting functions for Galerkin method and main function of Galerkin method without time variable
- fe1D_time.py -> main function for Galerkin method with time variable
- main.py -> Set arguments(constants and equations for GM) and call main function in fe1D_time.py

### Note
If you have small dx, you should set dt very very small. Refer to CFL condition.  
This would cause longer runtime due to larger nt for the same length of period.  
`total_time = dt * nt` 