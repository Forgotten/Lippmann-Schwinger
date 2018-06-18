# small scrip to compute the solution of Lippmann-Schwinger equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.

using PyPlot
using IterativeSolvers

include("../src/FastConvolution.jl")
include("../src/FastConvolution3D.jl")
include("../src/Preconditioner.jl")


# setting the number of threads for the FFT and BLAS
# libraries (please set them to match the number of
# physical cores in your system)
FFTW.set_num_threads(4);
BLAS.set_num_threads(4);

#Defining Omega
h = 1/64
k = 1/h

D0 = 0.0;

# size of box
a  = 1
x = collect(-a/2:h:a/2-h)
y = collect(-a/2:h:a/2-h)
z = collect(-a/2:h:a/2-h)
(n,m,l) = length(x), length(y), length(z)
N = n*m*l

X = zeros(n,m,l); Y = zeros(n,m,l); Z = zeros(n,m,l);
for i=1:n, j=1:m, p=1:l
    X[i,j,p] =  x[i];
    Y[i,j,p] =  y[j];
    Z[i,j,p] =  z[p];
end
X = X[:]; Y = Y[:]; Z = Z[:];
# we solve \triangle u + k^2(1 + nu(x))u = 0

# Defining the smooth perturbation of the slowness
nu(x,y,z) = 0.3*exp(-40*(x.^2 + y.^2 + z.^2)).*(abs(x).<0.48).*(abs(y).<0.48).*(abs(z).<0.48);

#
NU = reshape(nu(X,Y,Z), n,m,l);
#
figure(5); clf(); imshow(real(NU[:,:,15]))

## You can choose between Duan Rohklin trapezoidal quadrature
# fastconv = buildFastConvolution(x,y,h,k,nu)

# or Greengard Vico Quadrature (this is not optimized and is 2-3 times slower)
fastconv = buildFastConvolution3D(x,y,z,X,Y,Z,h,k,nu, quadRule = "Greengard_Vico");

# # assembling the sparsifiying preconditioner
@time As = buildSparseA3DConv(k,X,Y,Z,fastconv, n ,m, l);

# # assembling As*( I + k^2G*nu)
#Mapproxsp = As + k^2*(buildSparseAG3D(k,X,Y,Z,D0, n ,m,l)*spdiagm(nu(X,Y,Z)));
@time Mapproxsp = As + k^2*(buildSparseAG3DConv(k,X,Y,Z,fastconv, n ,m,l)*spdiagm(nu(X,Y,Z)));


# # defining the preconditioner
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

U = reshape(u+u_inc, n,m,l);

imshow(real(U[:,:,round(Integer, end/2)]))
