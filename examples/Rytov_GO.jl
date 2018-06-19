# small scrip to compute the solution of Lipman Schinwer equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.

using PyPlot
using IterativeSolvers
using SpecialFunctions


@everywhere include("../src/FastConvolution.jl")
@everywhere include("../src/Preconditioner.jl")

FFTW.set_num_threads(4);
BLAS.set_num_threads(4)

#Defining Omega
h = 0.001
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

spline(y) = (y.<0) +  (y.>=0).*(y.<1).*(2*y.^3 - 3*y.^2 + 1)
window(a1,b1,b2,a2,y) = (y.>=a1).*( (y.<b1).*spline(1/(abs(b1-a1)).*(-y+b1)) + (y.>=b1).*(y.<b2) + (y.>=b2).*spline(1/(abs(b2-a2)).*(y-b2)))

# Defining the smooth perturbation of the slowness
nu(x,y) = 0.05*sin(2*pi*x).*window(-0.48,-0.4,0.4,0.48,y).*window(-0.48,-0.05,0.05,0.48,x);

figure(1); clf();
imshow(real(reshape( (1-nu(X,Y)),n,m)), extent=[x[1], x[end], y[1], y[end]]);cb =  colorbar();


# Sampling the Green's function for the Toeplitz form
Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

fastconv = FastM(GFFT,nu(X,Y),2*n-1,2*m-1,n, m, k);

# assembling the sparsifiying preconditioner
As = buildSparseA(k,X,Y,D0, n ,m);

# assembling As*( I + k^2G*nu)
Mapproxsp = As + k^2*(buildSparseAG(k,X,Y,D0, n ,m)*spdiagm(nu(X,Y)));

# defining the preconditioner
precond = SparsifyingPreconditioner(Mapproxsp, As)

# building the RHS from the incident field
u_plane = exp(k*im*sqrt(1-nu(0*Y,X)).*X);

u_inc = exp(k*im*X);

rhs = -(fastconv*u_inc - u_inc);

# allocating the solution
u = zeros(Complex128,N);

# solving the system using GMRES
@time info =  gmres!(u, fastconv, rhs, Pl = precond)
# println(info[2].residuals[:])

# plotting the solution
figure(2)
clf()
imshow(real(reshape(u+u_inc,n,m)))

figure(3); clf()
imshow(real(reshape((u+u_inc)./u_inc, n,m))); colorbar()
title("\$u(x)/e^{ikx}\$ ")


intx = cumsum(sqrt(1-nu(x,0*x)))*h;
u_gem = exp(k*im*intx);

UUgem = repmat(u_gem, 1,n)

figure(4); clf()
imshow(real(reshape((u+u_inc)./UUgem[:], n,m))); colorbar()
title("\$u(x)/e^{ik \\phi (x)}\$ where \$ \\phi \$ comes from \$ \\int_{0}^{x} \\sqrt{1 + n(y)} dy \$ ")


refFact = 10;
xRef = -a/2:h/refFact:a/2

intxRef = cumsum(sqrt(1-nu(xRef,0*xRef)))*h/refFact;
u_gem = exp(k*im*intxRef)[1:refFact:end];

UUgem = repmat(u_gem, 1,n)

figure(14); clf()
imshow(real(reshape(-(u+u_inc)./UUgem[:], n,m))); colorbar()
title("\$u(x)/e^{ik \\phi (x)}\$ where \$ \\phi \$ comes from \$ \\int_{0}^{x} \\sqrt{1 + n(y)} dy \$ ")


UU = reshape(u+u_inc,n,m)
ucomp = UU[:,round(Integer,floor(end/2))]

# there is a two that had to disappear... I need to find the bug... :/
phi_1 = -1im*rhs./u_inc

figure(5); clf()
imshow(real(reshape(phi_1, n,m))); colorbar()
title("phase function \$phi_1(x)\$ ")

u_rytov = u_inc.*exp(1im*phi_1)

figure(6); clf()
imshow(real(reshape(u_rytov, n,m))); colorbar()
title("\$u_{rytov}\$" )

figure(7); clf()
imshow(real(reshape((u+u_inc)./ u_rytov, n,m))); colorbar()
title("\$u(x)/u_{rytov}(x)\$ ")


#Defining the smooth perturbation of the slowness
nu1(x,y) = nu(x,y).*window(-0.28,-0.10,0.10,0.28,y);

figure(24); clf();
imshow(real(reshape( (1-nu1(X,Y)),n,m)), extent=[x[1], x[end], y[1], y[end]]);cb =  colorbar();

fastconvWindow = FastM(GFFT,nu1(X,Y),2*n-1,2*m-1,n, m, k);

rhsWindow = -(fastconvWindow*u_inc - u_inc);


# there is a two that had to disappear... I need to find the bug... :/
phi_Window = -1im*rhsWindow./u_inc

figure(15); clf()
imshow(real(reshape(phi_1, n,m))); colorbar()
title("phase function phi_1(x) ")

u_rytovWindow = u_inc.*exp(1im*phi_Window)

figure(16); clf()
imshow(real(reshape(u_rytovWindow, n,m))); colorbar()
title("u_rytov ")

figure(17); clf()
imshow(real(reshape((u+u_inc)./ u_rytovWindow, n,m))); colorbar()
title("u/u_rytov ")


UU = reshape((u+u_inc)./u_rytov,n,m)
ucomp = UU[:,round(Integer,floor(end/2))]

figure(20); clf();
plot(real(ucomp[:]))



# u_inc = exp(k*im*X);

# rhs1 = -(fastconv1*u_inc - u_inc);

# # allocating the solution
# u1 = zeros(Complex128,N);

# # solving the system using GMRES
# @time info =  gmres!(u1, fastconv1, rhs1, precond1)
# println(info[2].residuals[:])

# # plotting the solution
# figure(5)
# clf()
# imshow(real(reshape(u1+u_inc,n,m)))

# figure(6); clf()
# imshow(real(reshape((u1+u_inc)./u_inc, n,m))); colorbar()

