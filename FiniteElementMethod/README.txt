- numint.py -> GaussLegendre polynomial for basis function
- fe1D_naive.py -> Galerkin method를 사용하기 위한 보조 함수와 time variable이 없는 경우의 Galerkin method
- fe1D_time.py -> time variable이 있는 경우의 Galerkin method
- main.py -> Galerkin Method에 들어갈 상수와 방정식을 세팅하고 실행하는 fe1D_time.py의 함수를 호출하는 파일

* Note
dx가 작은 경우 CFL condition에 의해 dt의 값도 매우매우 작아야합니다.
단, dt가 작아지는 경우, 같은 시간 시뮬레이션을 하기 위해서는 nt가 증가하며
전체 수행 시간이 길어질 수 있습니다.