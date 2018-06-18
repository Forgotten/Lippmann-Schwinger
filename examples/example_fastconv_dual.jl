# small scrip to compute the solution of Lippmann Schwinger equation
# We test the types introduced in FastConvolutionSlowGO.jl
# we test that the application is fast and that the construction
# is performed fast.
# we test a different formulation of the Lippmann-Schwinger equation

using PyPlot
using IterativeSolvers
using SpecialFunctions

include("../src/FastConvolutionDual.jl")
include("../src/FastConvolution.jl")
include("../src/Preconditioner.jl")

FFTW.set_num_threads(4);
BLAS.set_num_threads(4)

#Defining Omega
h = 2.0^(-8)
k = 200.0/4

# size of box
a = 1;
b = 1;
x = collect(-a/2:h:a/2)
y = collect(-b/2:h:b/2)
(n,m) = length(x), length(y)
N = n*m
X = repmat(x, 1, m)[:];
Y = repmat(y', n,1)[:];
# we solve \triangle u + k^2(1 + nu(x))u = 0

# We use the modified quadrature in Duan and Rohklin
(ppw,D) = referenceValsTrapRule();
D0 = D[1];


# window(y,alpha) = 1*(abs.(y).<=alpha/2) + (abs.(y).>alpha/2).*(abs.(y).<alpha).*exp.(2*exp.(-0.5*alpha./(abs.(y)-alpha/2))./ ((abs.(y)-alpha/2)./(0.5*alpha)-1) )

# window(y,alpha, beta) = 1*(abs.(y).<=beta) + (abs.(y).>beta).*(abs.(y).<alpha).*exp.(2*exp.(-(alpha- beta)./(abs.(y)-beta))./ ((abs.(y)-beta)./(alpha- beta)-1) )

# # Defining the smooth perturbation of the slowness
# nu(x,y) = 0.05*(1+sin(2*pi*x/(0.96))).*window(y,0.96*b/2, 0.48*b/2).*window(x,0.96*0.5, 0.3);

# # Defining the smooth perturbation of the slowness
 nu(x,y) = -0.3*exp.(-40*(x.^2 + y.^2)).*(abs.(x).<0.48).*(abs.(y).<0.48);

figure(1); clf();
imshow(real(reshape( (1 + nu(X,Y)),n,m)), extent=[y[1], y[end], x[1], x[end]]);cb =  colorbar();


###############################################################
###############################################################

# comparison with the true solution
fastconvDual = buildFastConvolutionDual(x,y,h,k,nu,quadRule ="Greengard_Vico" );
#fastconvDual = buildFastConvolutionDual(x,y,h,k,nu);

u_inc = exp.(k*im*X);

rhsDual = -k^2*nu(X,Y).*u_inc ;

# allocating the solution
sigma = zeros(Complex128,N);

# solving the system using GMRES
@time info =  gmres!(sigma, fastconvDual, rhsDual, log = true, maxiter = 10 )
# println(info[2].residuals[:])

figure(4)
clf()
imshow(real(reshape(sigma,n,m))); colorbar();
title("density \sigma")

uDual = FFTconvolution(fastconvDual,sigma);

figure(5)
clf()
imshow(real(reshape(uDual,n,m))); colorbar();
title("scattered wavefield")

figure(6)
clf()
imshow(real(reshape(uDual,n,m))); colorbar();

figure(7)
clf()
imshow(real(reshape(uDual+u_inc,n,m))); colorbar();


# testing the preconditioner (Not working yet )

@time (As, ADGs) = buildGSparseAandG(k,X,Y,D0, n ,m, nu);

Mapproxsp = -As + k^2*ADGs;

precondDual = SparsifyingPreconditioner(Mapproxsp, As);

# allocating the solution
sigmaprecond = zeros(Complex128,N);

# solving the system using GMRES
@time infoPrecond =  gmres!(sigmaprecond, fastconvDual, rhsDual,
                            Pl= precondDual, log = true)
# println(infoPrecond[2].residuals[:])

