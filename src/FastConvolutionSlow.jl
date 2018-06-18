# Class containing the objects for the windowed version of the integral operator
# for the slow varying quantity along the principal ray direction

type FastMslow
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    GFFT :: Array{Complex128,2}
    nu :: Array{Float64,1}
    x  :: Array{Float64,1}
    y  :: Array{Float64,1}
    # number of points in the extended domain
    ne :: Int64
    me :: Int64
    # number of points in the original domain
    n  :: Int64
    m  :: Int64
    # frequency
    omega :: Float64
    # direction of the plane wave
    e ::Array{Float64,1}
end

import Base.*
import Base.A_mul_B!
import Base.eltype
import Base.size

function size(M::FastMslow, dim::Int64)
  @assert dim<3 && dim>0
    return M.n*M.m
end

function *(M::FastMslow, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT

    # Allocate the space for the extended B
    BExt = zeros(Complex128,M.ne, M.me);
    # Apply spadiagm(nu) and ented by zeros
    BExt[1:M.n,1:M.m]= reshape((exp(1im*M.omega*(M.e[1]*M.x + M.e[2]*M.y)).*M.nu).*b,M.n,M.m) ;

    # Fourier Transform
    BFft = fft(BExt)
    # Component-wise multiplication
    BFft = M.GFFT.*BFft
    # Inverse Fourier Transform
    BExt = ifft(BFft)

    # multiplication by omega^2
    B = M.omega^2*(BExt[M.n:2*M.n-1, M.m:2*M.m-1]);

    return (b + (exp(-1im*M.omega*(M.e[1]*M.x + M.e[2]*M.y)).*(B[:])))
end

function A_mul_B!(Y,
                  M::FastMslow,
                  V)
    # in place matrix matrix multiplication
    @assert(size(Y) == size(V))
    # print(size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = M*V[:,ii]
    end
end

# # TO DO: add more parameters for the window in here
# function buildFastConvolutionSlow(x::Array{Float64,1},y::Array{Float64,1},
#                                   h::Float64,k,nu::Function, e::Array{Float64,1};
#                                   quadRule::String = "trapezoidal",
#                                   window::String = "normal")

#     if quadRule == "trapezoidal"
#         if window == "normal"

#             (ppw,D) = referenceValsTrapRule();
#             D0      = D[round(Int,k*h)];
#             (n,m) = length(x), length(y)
#             Ge    = buildGConv(x,y,h,n,m,D0,k);
#             GFFT  = fft(Ge);
#             X = repmat(x, 1, m)[:]
#             Y = repmat(y', n,1)[:]
#         end
#     return FastM(GFFT,nu(X,Y),2*n-1,2*m-1,n, m, k);

#   elseif quadRule == "Greengard_Vico"

#       Lp = 4*(abs(x[end] - x[1]) + h)
#       L  =   (abs(x[end] - x[1]) + h)*1.5
#       (n,m) = length(x), length(y)
#       X = repmat(x, 1, m)[:]
#       Y = repmat(y', n,1)[:]

#       # this is depending if n is odd or not
#       if mod(n,2) == 0
#         kx = (-(2*n):1:(2*n-1));
#         ky = (-(2*m):1:(2*m-1));

#         KX = (2*pi/Lp)*repmat(kx, 1, 4*m);
#         KY = (2*pi/Lp)*repmat(ky', 4*n,1);

#         S = sqrt(KX.^2 + KY.^2);

#         GFFT = Gtruncated2D(L, k, S)
#         return FastM(GFFT, nu(X,Y), 4*n, 4*m,
#                      n, m, k , quadRule="Greengard_Vico");
#       else
#         # kx = (-2*(n-1):1:2*(n-1) )/4;
#         # ky = (-2*(m-1):1:2*(m-1) )/4;

#         # KX = (2*pi/Lp)*repmat(kx, 1, 4*m-3);
#         # KY = (2*pi/Lp)*repmat(ky', 4*n-3,1);

#         # S = sqrt(KX.^2 + KY.^2);

#         # GFFT = Gtruncated2D(L, k, S)

#         # return FastM(GFFT,nu(X,Y),4*n-3,4*m-3,
#         #              n,m, k,quadRule = "Greengard_Vico");

#         kx = (-2*n:1:2*n-1);
#         ky = (-2*m:1:2*m-1);

#         KX = (2*pi/Lp)*repmat( kx, 1,4*m);
#         KY = (2*pi/Lp)*repmat(ky',4*n,  1);

#         S = sqrt(KX.^2 + KY.^2);

#         GFFT = Gtruncated2D(L, k, S)

#         return FastM(GFFT,nu(X,Y),4*n,4*m,
#                      n,m, k,quadRule = "Greengard_Vico");


#     end
#   end
# end



function buildGConvWindowed(x,y,h::Float64,n::Int64,m::Int64,D0,k::Float64)
    # function to build the convolution vector for the
    # fast application of the convolution, in this case we add a directional
    # filter to eliminate the oscillations

    # this is built for odd n and odd m.

    if mod(n,2) == 1
      # build extended domain
      xe = collect((x[1]-(n-1)/2*h):h:(x[end]+(n-1)/2*h));
      ye = collect((y[1]-(m-1)/2*h):h:(y[end]+(m-1)/2*h));

      Xe = repmat(xe, 1, 2*m-1);
      Ye = repmat(ye', 2*n-1,1);
      # to avoid evaluating at the singularity
      indMiddle = m

    else

      println("so far only works for n odd")
      # to be done
      # # build extended domain
      # xe = collect((x[1]-n/2*h):h:(x[end]+n/2*h));
      # ye = collect((y[1]-m/2*h):h:(y[end]+m/2*h));

      # Xe = repmat(xe, 1, 2*m-1);
      # Ye = repmat(ye', 2*n-1,1);
      # # to avoid evaluating at the singularity
      # indMiddle = m

    end

    R = sqrt(Xe.^2 + Ye.^2);

    # we modify R to remove the zero (so we don't )
    R[indMiddle,indMiddle] = 1;
    # sampling the Green's function
    Ge = sampleGkernelpar(k,R,h)
    #Ge = pmap( x->1im/4*hankelh1(0,k*x)*h^2, R)
    # modiyfin the diagonal with the quadrature
    # modification
    Ge[indMiddle,indMiddle] = 1im/4*D0*h^2;

    return Ge

end
