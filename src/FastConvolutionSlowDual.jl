
type FastMslowDual
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
    e :: Array{Float64,1}
    quadRule :: String
    function FastMslowDual(GFFT::Array{Complex128,2},
                           nu::Array{Float64,1},
                           x::Array{Float64,1},
                           y::Array{Float64,1},
                           ne::Int64,me::Int64,n::Int64,m::Int64,
                           k::Float64,e::Array{Float64,1}; quadRule::String = "trapezoidal")
      return new(GFFT,nu,x,y,ne,me,n, m, k,e, quadRule)
    end
end


import Base.*

function *(M::FastMslowDual, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT
    # dummy function to call fastconvolution
    return fastconvolution(M,b)
end


@inline function fastconvolution(M::FastMslowDual, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT

    # computing omega^2 nu G*(b)
    B = M.omega^2*(M.nu.*FFTconvolution(M,exp(1im*M.omega*(M.e[1]*M.x + M.e[2]*M.y)).*b))

    # returning b + G*(b nu)
    return (-b + exp(-1im*M.omega*(M.e[1]*M.x + M.e[2]*M.y)).*B[:])
end


@inline function FFTconvolution(M::FastMslowDual, b::Array{Complex128,1})
    # function to overload the applyication of
    # convolution of b times G

    if M.quadRule == "trapezoidal"

      #obtaining the middle index
      indMiddle = round(Integer, M.n)

      # Allocate the space for the extended B
      BExt = zeros(Complex128,M.ne, M.me);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.m]= reshape(M.nu.*b,M.n,M.m) ;

      # Fourier Transform
      BFft = fft(BExt)
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(BFft)

      # multiplication by omega^2
      B = (BExt[M.n:M.n+M.n-1, M.m:M.m+M.m-1]);

    elseif M.quadRule == "Greengard_Vico"
      # for this we use the Greengard Vico method in the
      # frequency domain

      # Allocate the space for the extended B
      BExt = zeros(Complex128,M.ne, M.me);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.m]= reshape(M.nu.*b,M.n,M.m) ;

      # Fourier Transform
      BFft = fftshift(fft(BExt)) 
     # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(ifftshift(BFft))

      # multiplication by omega^2
      B = BExt[1:M.n, 1:M.m];
    end
    return  B[:]
end


function buildFastConvolutionSlowDual(x::Array{Float64,1},y::Array{Float64,1},
                                      h::Float64,k::Float64,
                                      nu::Function, 
                                      e::Array{Float64,1}; 
                                      quadRule::String = "trapezoidal")

  if quadRule == "trapezoidal"

    (ppw,D) = referenceValsTrapRule();
    D0      = D[round(Int,k*h)];
    (n,m) = length(x), length(y)
    Ge    = buildGConv(x,y,h,n,m,D0,k);
    GFFT  = fft(Ge);
    X = repmat(x, 1, m)[:]
    Y = repmat(y', n,1)[:]

    return FastMslowDual(GFFT,nu(X,Y),X,Y,2*n-1,2*m-1,n, m, k, e);

  elseif quadRule == "Greengard_Vico"

      Lp = 4*(abs(x[end] - x[1]) + h)
      L  =   (abs(x[end] - x[1]) + h)*1.5
      (n,m) = length(x), length(y)
      X = repmat(x, 1, m)[:]
      Y = repmat(y', n,1)[:]

      # this is depending if n is odd or not
      if mod(n,2) == 0
        kx = (-(2*n):1:(2*n-1));
        ky = (-(2*m):1:(2*m-1));

        KX = (2*pi/Lp)*repmat(kx, 1, 4*m);
        KY = (2*pi/Lp)*repmat(ky', 4*n,1);

        S = sqrt(KX.^2 + KY.^2);

        GFFT = Gtruncated2D(L, k, S)
        return FastMslowDual(GFFT, nu(X,Y),X,Y, 4*n, 4*m,
                     n, m, k , e,quadRule="Greengard_Vico");
      else
        kx = (-2*n:1:2*n-1);
        ky = (-2*m:1:2*m-1);

        KX = (2*pi/Lp)*repmat( kx, 1,4*m);
        KY = (2*pi/Lp)*repmat(ky',4*n,  1);

        S = sqrt(KX.^2 + KY.^2);

        GFFT = Gtruncated2D(L, k, S)

        return FastMslowDual(GFFT,nu(X,Y),X,Y,4*n,4*m,
                     n,m, k,e,quadRule = "Greengard_Vico");


    end
  end
end
