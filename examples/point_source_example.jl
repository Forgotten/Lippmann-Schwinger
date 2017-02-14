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
h = 0.002
k = 2*pi*

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

xs = -0.4; ys = -0.4;                     # point source location
sigma = 0.15;

window(y,alpha, beta) = 1*(abs(y).<=beta) + (abs(y).>beta).*(abs(y).<alpha).*exp(2*exp(-(alpha- beta)./(abs(y)-beta))./ ((abs(y)-beta)./(alpha- beta)-1) ) 

# version for Jun
# xHet = 0.1;
# yHet = 0.1;
# # Defining the smooth perturbation of the slowness
# nu(x,y) = -0.5*exp( -1/(2*sigma^2)*((x-xHet).^2 + (y-yHet).^2) ).*window(sqrt((x-xHet).^2 + (y-yHet).^2), 0.3,0.38  );

#version for Matthias 
xHet = [ -0.131 -0.055  0.052 -0.399 -0.082 -0.170  0.253  -0.192   0.186  0.139 -0.080  0.113  0.108 -0.235  0.178  0.052  0.179 -0.178  0.244  0.020 ];
yHet = [ -0.417 -0.103  0.371  0.384  0.300  0.121 -0.026   0.352  -0.054 -0.455 -0.262  0.080 -0.053  0.364 -0.210  0.349  0.210 -0.247  0.142 -0.309 ];

sigma = [ 0.162  0.1295 0.175 0.143  0.089  0.169  0.195  0.188  0.101  0.119  0.0365  0.052 0.059 0.182  0.090 0.174853  0.082 0.0942473 0.060431 0.1848739
];
amplitude = 5*[-0.0286571 -0.0422643  0.0761454 -0.0067917 -0.0538421  0.0347438  0.070426  -0.0577631 -0.0806493  0.0262168  0.014281  0.079996 -0.060261 -0.083327  0.086252 -0.041758 -0.089415  0.085001 -0.096065 -0.0828878
];

# Defining the smooth perturbation of the slowness
nu(x,y) =  (amplitude[1]*exp( -1/(2*sigma[1]^2)*((x-xHet[1]).^2 + (y-yHet[1]).^2) )+
            amplitude[2]*exp( -1/(2*sigma[2]^2)*((x-xHet[2]).^2 + (y-yHet[2]).^2) )+
            amplitude[3]*exp( -1/(2*sigma[3]^2)*((x-xHet[3]).^2 + (y-yHet[3]).^2) )+
            amplitude[4]*exp( -1/(2*sigma[4]^2)*((x-xHet[4]).^2 + (y-yHet[4]).^2) )+
            amplitude[5]*exp( -1/(2*sigma[5]^2)*((x-xHet[5]).^2 + (y-yHet[5]).^2) )+
            amplitude[6]*exp( -1/(2*sigma[6]^2)*((x-xHet[6]).^2 + (y-yHet[6]).^2) )+
            amplitude[7]*exp( -1/(2*sigma[7]^2)*((x-xHet[7]).^2 + (y-yHet[7]).^2) )+
            amplitude[8]*exp( -1/(2*sigma[8]^2)*((x-xHet[8]).^2 + (y-yHet[8]).^2) )+
            amplitude[9]*exp( -1/(2*sigma[9]^2)*((x-xHet[9]).^2 + (y-yHet[9]).^2) )+
            amplitude[10]*exp( -1/(2*sigma[10]^2)*((x-xHet[10]).^2 + (y-yHet[10]).^2) )+
            amplitude[11]*exp( -1/(2*sigma[11]^2)*((x-xHet[11]).^2 + (y-yHet[11]).^2) )+
            amplitude[12]*exp( -1/(2*sigma[12]^2)*((x-xHet[12]).^2 + (y-yHet[12]).^2) )+
            amplitude[13]*exp( -1/(2*sigma[13]^2)*((x-xHet[13]).^2 + (y-yHet[13]).^2) )+
            amplitude[14]*exp( -1/(2*sigma[14]^2)*((x-xHet[14]).^2 + (y-yHet[14]).^2) )+
            amplitude[15]*exp( -1/(2*sigma[15]^2)*((x-xHet[15]).^2 + (y-yHet[15]).^2) )+
            amplitude[16]*exp( -1/(2*sigma[16]^2)*((x-xHet[16]).^2 + (y-yHet[16]).^2) )+
            amplitude[17]*exp( -1/(2*sigma[17]^2)*((x-xHet[17]).^2 + (y-yHet[17]).^2) )+
            amplitude[18]*exp( -1/(2*sigma[18]^2)*((x-xHet[18]).^2 + (y-yHet[18]).^2) )+
            amplitude[19]*exp( -1/(2*sigma[19]^2)*((x-xHet[19]).^2 + (y-yHet[19]).^2) )+
            amplitude[20]*exp( -1/(2*sigma[20]^2)*((x-xHet[20]).^2 + (y-yHet[20]).^2) ) ).*(window(abs(x),0.48,0.4 ) ).*(window(abs(y),0.48,0.4 ) ).*
            (1 - window(sqrt( (x-xs).^2 + (y-ys).^2),0.1,0.05 ) );
       

figure(1);clf();
imshow(reshape(sqrt(1 - nu(X,Y)), n,m),extent=[y[1], y[end], x[end], x[1]]);cb =  colorbar();


## You can choose between Duan Rohklin trapezoidal quadrature
fastconv = buildFastConvolution(x,y,h,k,nu)

# or Greengard Vico Quadrature (this is not optimized and it is 2-3 times slower)
#fastconv = buildFastConvolution(x,y,h,k,nu, quadRule = "Greengard_Vico");

# assembling the sparsifiying preconditioner
@time As = buildSparseA(k,X,Y,D0, n ,m);

# assembling As*( I + k^2G*nu)
@time Mapproxsp = As + k^2*(buildSparseAG(k,X,Y,D0, n ,m)*spdiagm(nu(X,Y)));

# defining the preconditioner
precond = SparsifyingPreconditioner(Mapproxsp, As)

# building the RHS from the incident field
u_inc = 1im/4*hankelh1(0, k*sqrt((X-xs).^2 + (Y-ys).^2)+eps(1.0));
rhs = -k^2*FFTconvolution(fastconv, nu(X,Y).*u_inc) ;
figure(2)
clf()
imshow(imag(reshape(rhs,n,m)))
#rhs = u_inc;

# allocating the solution
u = zeros(Complex128,N);

# solving the system using GMRES
@time info =  gmres!(u, fastconv, rhs, precond)
println(info[2].residuals[:])

# plotting the solution
figure(3)
clf()
imshow(imag(reshape(u+u_inc,n,m)))
