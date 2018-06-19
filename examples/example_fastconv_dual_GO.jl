# small scrip to compute the solution of Lippmann Schwinger equation
# We test that the density sigma can be properly factorized as a
# mu_slow*e^{i \omega \phi(x)} where $\phi$ accounts for the travel
# time within the geometrical optics solution

using PyPlot
using IterativeSolvers
using SpecialFunctions

include("../src/FastConvolutionDual.jl")
include("../src/FastConvolution.jl")
include("../src/Preconditioner.jl")
include("../src/FastConvolutionSlowDual.jl")


FFTW.set_num_threads(4);
BLAS.set_num_threads(4)

#Defining Omega
h = 0.0015
k = 1/h

# size of box
a = 1;
b = 0.125;
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


window(y,alpha) = 1*(abs.(y).<=alpha/2) + (abs.(y).>alpha/2).*(abs.(y).<alpha).*
                     exp.(2*exp.(-0.5*alpha./(abs.(y)-alpha/2))./
                         ((abs.(y)-alpha/2)./(0.5*alpha)-1) )

window(y,alpha, beta) = 1*(abs.(y).<=beta) + (abs.(y).>beta).*(abs.(y).<alpha).*
                           exp.(2*exp.(-(alpha- beta)./(abs.(y)-beta))./
                               ((abs.(y)-beta)./(alpha- beta)-1) )

# Defining the smooth perturbation of the slowness
nu(x,y) = -0.05*(sin.(4*pi*x/(0.96))).*
          window(y,0.96*b/2, 0.48*b/2).*
          window(x,0.96*0.5, 0.3);


figure(1); clf();
imshow(real(reshape( (1 + nu(X,Y)),n,m)),
       extent=[y[1], y[end], x[1], x[end]]); cb =  colorbar();


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
@time info =  gmres!(sigma, fastconvDual, rhsDual, maxiter = 10 )
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

figure(7)
clf()
imshow(real(reshape(uDual+u_inc,n,m))); colorbar();
title("total wavefield")


# computing the geometric optic phi
phiGO = cumsum(reshape(sqrt(1+nu(X,Y)), n,m))*h-0.5;
u_gem = exp(k*im*phiGO);

# plotting the solution
figure(8)
clf()
imshow(real(reshape(sigma./u_gem[:],n,m)))
title("sigma/ u_geom ")

figure(9)
clf()
imshow(real(reshape(sigma./u_inc[:],n,m))); colorbar();
title("sigma/ u_inc ")

figure(10)
clf()
imshow(real(reshape(uDual./u_gem[:],n,m)))
title("u / u_geom ")



fastconvSlowDual = buildFastConvolutionSlowDual(x,y,h,k,nu, [1.0, 0.0],
                                                quadRule ="Greengard_Vico");

rhsSlowDual = -k^2*nu(X,Y) + zeros(Complex128,N);

# allocating the solution
sigmaSlow = zeros(Complex128,N);

# solving the system using GMRES
@time info =  gmres!(sigmaSlow, fastconvSlowDual, rhsSlowDual, maxiter = 10 )
# println(info[2].residuals[:])


figure(11)
clf()
imshow(real(reshape(sigmaSlow,n,m)))
title("sigma Slow")


norm2 = maximum(abs(sigma./u_inc[:]))
figure(21)
clf()
imshow(abs(reshape(sigmaSlow - sigma./u_inc[:],n,m)/norm2)); colorbar();
title("error")
