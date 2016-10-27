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
hInit = 1/8;
nInit = 8;
k = 18.0
uRho = []

sigma = 0.05
renorm_factor = 1/(sigma^2*(2*pi));
rho(x,y) = renorm_factor*exp((-1/(2*sigma^2))*(x.^2 + y.^2 ));

nu(x,y) = 0.3*exp(-40*(x.^2 + y.^2)).*(abs(x).<0.48).*(abs(y).<0.48);

nSamples = 8

for ii = 1:nSamples

    #h even 
    h = hInit/2.0^((ii-1))
    # h odd
    #h = 1/(nInit*2.0^((ii-1))+1)

    # size of box
    a = 1
    # x = collect(-a/2:h:a/2)
    # y = collect(-a/2:h:a/2)
    x = collect(-a/2:h:a/2-h)
    y = collect(-a/2:h:a/2-h)
    (n,m) = length(x), length(y)
    println(n)
    N = n*m
    X = repmat(x, 1, m)[:]
    Y = repmat(y', n,1)[:]
    # we solve \triangle u + k^2(1 + nu(x))u = 0
    

    Rho = rho(X,Y) + 0*1im;
    RHO = reshape(Rho, n,m);
    
    figure(5); clf();
    imshow(real(RHO))



    # Greengard Vico Quadrature (this is not optimized and is 2-3 times slower)
    fastconv = buildFastConvolution(x,y,h,k,nu, quadRule = "Greengard_Vico");

    u_rho = FFTconvolution(fastconv, Rho);
    u_rho = reshape(u_rho,n,m)

    u_inc = exp(k*im*X);
    u_conv = FFTconvolution(fastconv, u_inc );
    u_conv = reshape(u_conv,n,m)   


    push!(uRho, u_rho[1:round(Integer,2.0^((ii-1))):end, 1:round(Integer,2.0^((ii-1))):end])
    
end

error = zeros(nSamples-1)
for ii = 1:nSamples-1
    error[ii] = norm( uRho[end][:] - uRho[ii][:])/norm(uRho[end][:])
end



println(error)