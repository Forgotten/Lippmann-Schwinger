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
#Defining Omega
h = 1/32
k = 2.0

# size of box
a  = 1
x = collect(-a/2:h:a/2-h)
y = collect(-a/2:h:a/2-h)
z = collect(-a/2:h:a/2-h)

(n,m,l) = length(x), length(y), length(z)

N = n*m*l

# creating the 3D tensor grid
# TODO: write a
X = zeros(n,m,l); Y = zeros(n,m,l); Z = zeros(n,m,l);
for i=1:n, j=1:m, p=1:l
    X[i,j,p] =  x[i];
    Y[i,j,p] =  y[j];
    Z[i,j,p] =  z[p];
end

X = X[:]; Y = Y[:]; Z = Z[:];

# Defining the smooth perturbation of the slowness
sigma = 0.05
renorm_factor = 1/(sigma^3*(2*pi)^(3/2));
rho(x,y,z) = renorm_factor*exp((-1/(2*sigma^2))*(x.^2 + y.^2 + z.^2));

Rho = rho(X,Y,Z) + 0*1im;
RHO = reshape(Rho, n,m,l);

figure(5); clf();
imshow(real(RHO[:,:,round(Integer, end/2)]))
## You can choose between Duan Rohklin trapezoidal quadrature
# fastconv = buildFastConvolution(x,y,h,k,nu)

# defining a dummy nu
nu(x,y,z) = 1*exp(-40*(x.^2 + y.^2 + z.^2)).*(abs(x).<0.48).*(abs(y).<0.48).*(abs(z).<0.48);

# or Greengard Vico Quadrature (this is not optimized and is 2-3 times slower)
fastconv = buildFastConvolution3D(x,y,z,X,Y,Z,h,k,nu, quadRule = "Greengard_Vico");

u_rho = FFTconvolution(fastconv, Rho);

solRef(x,y,z) = exp(-sigma^2*k^2/2)./(4*pi*sqrt(x.^2+y.^2+z.^2)).*
                (real(exp(-1im*k*sqrt(x.^2+y.^2+z.^2)).*erf((2*sigma^2*1im*k - 2*sqrt(x.^2 + y.^2 + z.^2))/(2*sqrt(2*sigma^2)))) - 1im*sin(k*sqrt(x.^2 + y.^2 + z.^2)))

uRef = solRef(X,Y,Z);

# plotting the solution
figure(1)
clf()
UU = reshape(u_rho,n,m,l)
imshow(real(UU[:,:,round(Integer, end/2)])); colorbar();

figure(2)
clf();
UURef = reshape(uRef ,n,m,l)
imshow(real(UURef[:,:,round(Integer, end/2)])); colorbar();


figure(3)
clf();
imshow(abs((UURef-UU)[:,:,round(Integer, end/2)])); colorbar()
println("Error in L infinity norm is ", maximum(abs(u_rho - uRef) ))

