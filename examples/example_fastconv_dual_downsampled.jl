# small scrip to compute the solution of Lipman Schinwer equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.


using PyPlot
using IterativeSolvers

include("../src/FastConvolutionDual.jl")
include("../src/FastConvolution.jl")
include("../src/Preconditioner.jl")
include("../src/FastConvolutionSlowDual.jl")
include("../src/FastConvolutionDualDownSampled.jl")


FFTW.set_num_threads(4);
BLAS.set_num_threads(4)

#Defining Omega
h = 0.00025
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

downsample=4;

window(y,alpha) = 1*(abs(y).<=alpha/2) + (abs(y).>alpha/2).*(abs(y).<alpha).*
                     exp(2*exp(-0.5*alpha./(abs(y)-alpha/2))./
                         ((abs(y)-alpha/2)./(0.5*alpha)-1) )

window(y,alpha, beta) = 1*(abs(y).<=beta) + (abs(y).>beta).*(abs(y).<alpha).*
                           exp(2*exp(-(alpha- beta)./(abs(y)-beta))./
                               ((abs(y)-beta)./(alpha- beta)-1) )

# Defining the smooth perturbation of the slowness
nu(x,y) = -0.05*(sin(4*pi*x/(0.96))).*
          window(y,0.96*b/2, 0.48*b/2).*
          window(x,0.96*0.5, 0.3);


figure(1); clf();
imshow(real(reshape( (1 + nu(X,Y)),n,m)),
       extent=[y[1], y[end], x[1], x[end]]); cb =  colorbar();



fastconvSlowDual = buildFastConvolutionSlowDual(x,y,h,k,nu, [1.0, 0.0],
                                                quadRule ="Greengard_Vico");

rhsSlowDual = -k^2*nu(X,Y) + zeros(Complex128,N);

# allocating the solution
sigmaSlow = zeros(Complex128,N);

# solving the system using GMRES
@time info =  gmres!(sigmaSlow, fastconvSlowDual, rhsSlowDual, maxiter = 10 )
println(info[2].residuals[:])

using Interpolations


fastconvSlowDualDown = FastDualDownSampled(fastconvSlowDual, downsample)



index1 = 1:downsample:n
index2 = 1:downsample:m

Sindex = spzeros(n,m)
for ii = 0:round(Integer,(n-1)/downsample)
    for jj = 0:round(Integer,(m-1)/downsample)
        Sindex[1+ii*downsample,1+jj*downsample ] = 1
    end
end

index = find(Sindex[:])
S = speye(n*m, n*m)
Sampling = S[index,:];

sigmaSlowDown = zeros(Complex128, size(Sampling)[1])

@time info =  gmres!(sigmaSlowDown, fastconvSlowDualDown, Sampling*rhsSlowDual, maxiter = 10 )
println(info[2].residuals[:])


U_Window = reshape(uslowWindow, n,m)
u_downsampled = Sampling*uslowWindow;

sigmaDownsampled = reshape(sigmaSlowDown, round(Integer,(n-1)/downsample) +1, round(Integer,(m-1)/downsample) +1)

figure(17);
clf();
imshow(real(sigmaDownsampled))
title("sigma Slow in coarse mesh")


knots = (collect(1:downsample:n), collect(1:downsample:m))

itp_real = interpolate(knots, real(sigmaDownsampled), Gridded(Linear()))
itp_imag = interpolate(knots, imag(sigmaDownsampled), Gridded(Linear()))

interpsigma = itp_real[collect(1:n),collect(1:m) ] + 1im*itp_imag[collect(1:n),collect(1:m) ]

figure(15);
clf();
imshow(real(interpsigma))
title("interpolated sigma slow")


normSigmaSlow = maximum(abs(sigmaSlow))

figure(15);
clf();
imshow(real(interpsigma - reshape(sigmaSlow,n,m))/normSigmaSlow)
colorbar()
title("relative error");

FastDown =  FastDownSampled(fastconvslowWindow, downsample)

uDown = zeros(Complex128, round(Integer,(m-1)/downsample +1).^2)

@time info =  gmres!(uDown, FastDown, Sampling*rhsslowWindow )

println(info[2].residuals[:])
