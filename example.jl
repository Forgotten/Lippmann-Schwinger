# small scrip to compute the solution of Lipman Schinwer equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.

using PyPlot
using IterativeSolvers

include("FastConvolution.jl")
#Defining Omega
h = 0.005
k = 1/h

# size of box
a  = 1
x = -a/2:h:a/2
y = -a/2:h:a/2
(n,m) = length(x), length(y)
N = n*m
X = repmat(x, 1, m)[:]
Y = repmat(y', n,1)[:]
# we solve \triangle u + k^2(1 + nu(x))u = 0

# We use the modified quadrature in Ruan and Rohklin
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

# Defining the smooth perturbation of the slowness
nu(x,y) = -0.3*exp(-40*(x.^2 + y.^2)).*(abs(x).<0.48).*(abs(y).<0.48);

# Sampling the Green's function for the Toeplitz form
Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);

# assembling the sparsifiying preconditioner
As = buildSparseA(k,X,Y,D0, n ,m);

# assembling As*( I + k^2G*nu)
Mapproxsp = As + k^2*(buildSparseAG(k,X,Y,D0, n ,m)*spdiagm(nu(X,Y)));

# Lu factorization (via multifrontal method using UMFPACK)
Minv = lufact(Mapproxsp);

# defining the preconditioned system
precond(x) = (Minv\(As*(fastconv*x)));

# building the RHS from the incident field
u_inc = exp(k*im*X);
rhs = -(fastconv*u_inc - u_inc);

# allocating the solution
u = zeros(Complex128,N);

# solving the system using GMRES
@time info =  gmres!(u, precond, Minv\(As*rhs))
println(info[2].residuals[:])

figure(1)
clf()
imshow(real(reshape(u+u_inc,n,m)))

