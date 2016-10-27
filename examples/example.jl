# small scrip to compute the solution of Lippmann-Schwinger equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.

using PyPlot
using IterativeSolvers

include("../src/FastConvolution.jl")
include("../src/Preconditioner.jl")


# setting the number of threads for the FFT and BLAS
# libraries (please set them to match the number of
# physical cores in your system)
FFTW.set_num_threads(4);
BLAS.set_num_threads(4);


#Defining Omega
h = 0.005
k = 1/(h)

# size of box
a = 1
x = collect(-a/2:h:a/2)
y = collect(-a/2:h:a/2)
(n,m) = length(x), length(y)
N = n*m
X = repmat(x, 1, m)[:]
Y = repmat(y', n,1)[:]
# we solve \triangle u + k^2(1 + nu(x))u = 0

# We use the modified quadrature in Ruan and Rohklin
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

# Defining the smooth perturbation of the slowness
nu(x,y) = 0.3*exp(-40*(x.^2 + y.^2)).*(abs(x).<0.48).*(abs(y).<0.48);

## You can choose between Duan Rohklin trapezoidal quadrature
# fastconv = buildFastConvolution(x,y,h,k,nu)

# or Greengard Vico Quadrature (this is not optimized and is 2-3 times slower)
fastconv = buildFastConvolution(x,y,h,k,nu, quadRule = "Greengard_Vico");

# assembling the sparsifiying preconditioner
As = buildSparseA(k,X,Y,D0, n ,m);

# assembling As*( I + k^2G*nu)
Mapproxsp = As + k^2*(buildSparseAG(k,X,Y,D0, n ,m)*spdiagm(nu(X,Y)));

# defining the preconditioner
precond = SparsifyingPreconditioner(Mapproxsp, As)

# building the RHS from the incident field
u_inc = exp(k*im*X);
rhs = -(fastconv*u_inc - u_inc);

# allocating the solution
u = zeros(Complex128,N);

# solving the system using GMRES
@time info =  gmres!(u, fastconv, rhs, precond)
println(info[2].residuals[:])

# plotting the solution
figure(1)
clf()
imshow(real(reshape(u+u_inc,n,m)))
