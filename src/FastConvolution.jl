# File with the functions necessary to implement the
# sparsifiying preconditioner
# Ying 2014 Sparsifying preconditioners for the Lippmann-Schwinger Equation

include("Functions.jl")

type FastM3D
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    GFFT :: Array{Complex128,3}
    nu :: Array{Float64,1}
    # number of points in the extended domain
    ne :: Int64
    me :: Int64
    le :: Int64
    # number of points in the original domain
    n  :: Int64
    m  :: Int64
    l  :: Int64
    # frequency
    omega :: Float64
    quadRule :: String
    function FastM3D(GFFT,nu,ne,me,le,n,m,l,k;quadRule::String = "Greengard_Vico")
      return new(GFFT,nu,ne,me,le,n,m,l,k,quadRule)
    end
end

type FastM
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    GFFT :: Array{Complex128,2}
    nu :: Array{Float64,1}
    # number of points in the extended domain
    ne :: Int64
    me :: Int64
    # number of points in the original domain
    n  :: Int64
    m  :: Int64
    # frequency
    omega :: Float64
    quadRule :: String
    function FastM(GFFT,nu,ne,me,n,m,k; quadRule::String = "trapezoidal")
      return new(GFFT,nu,ne,me,n, m, k, quadRule)
    end
end

import Base.*

function *(M::FastM, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT
    # dummy function to call fastconvolution
    return fastconvolution(M,b)
end


function *(M::FastM3D, b::Array{Complex128,1}; verbose::Bool=false)
    # multiply by nu, compute the convolution and then
    # multiply by omega^2
    B = M.omega^2*(FFTconvolution(M,M.nu.*b, verbose=verbose))

    return (b + B)
end



@inline function FFTconvolution(M::FastM3D, b::Array{Complex128,1};
                                verbose::Bool=false )
    verbose && println("Application of the 3D convolution")
    # Allocate the space for the extended B
    BExt = zeros(Complex128,M.ne, M.ne, M.le);
    # zero padding
    BExt[1:M.n,1:M.m,1:M.l]= reshape(b,M.n,M.m,M.l) ;

    # Fourier Transform
    BFft = fftshift(fft(BExt))
    # Component-wise multiplication
    BFft = M.GFFT.*BFft
    # Inverse Fourier Transform
    BExt = ifft(ifftshift(BFft))

    # multiplication by omega^2
    B = BExt[1:M.n,1:M.m,1:M.l];

    return B[:]
end

@inline function fastconvolution(M::FastM, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT

    # computing G*(b nu)

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
      B = M.omega^2*(BExt[M.n:M.n+M.n-1, M.m:M.m+M.m-1]);

    elseif M.quadRule == "Greengard_Vico"
      # for this we use the Greengard Vico method in the
      # frequency domain

      # Allocate the space for the extended B
      BExt = zeros(Complex128,M.ne, M.me);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.n]= reshape(M.nu.*b,M.n,M.m) ;

      # Fourier Transform
      BFft = fftshift(fft(BExt))
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(ifftshift(BFft))

      # multiplication by omega^2
      B = M.omega^2*(BExt[1:M.n, 1:M.m]);

    end

    # returning b + G*(b nu)
    return (b + B[:])
end

@inline function FFTconvolution(M::FastM, b::Array{Complex128,1})
    # function to overload the applyication of
    # convolution of b times G

    if M.quadRule == "trapezoidal"

      #obtaining the middle index
      indMiddle = round(Integer, M.n)

      # Allocate the space for the extended B
      BExt = zeros(Complex128,M.ne, M.ne);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.n]= reshape(b,M.n,M.n) ;

      # Fourier Transform
      BFft = fft(BExt)
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(BFft)

      # multiplication by omega^2
      B = (BExt[indMiddle: indMiddle+M.n-1, indMiddle:indMiddle+M.n-1]);

    elseif M.quadRule == "Greengard_Vico"
      # for this we use the Greengard Vico method in the
      # frequency domain

      # Allocate the space for the extended B
      BExt = zeros(Complex128,M.ne, M.ne);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.n]= reshape(b,M.n,M.n) ;

      # Fourier Transform
      BFft = fftshift(fft(BExt))
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(ifftshift(BFft))

      # multiplication by omega^2
      B = (BExt[1:M.n, 1:M.n]);

    end
    return  B[:]
end


# #this is the sequantial version for sampling G
# function sampleG(k,X,Y,indS, D0)
#     # function to sample the Green's function at frequency k
#     Gc = zeros(Complex128, length(indS), length(X))
#     for i = 1:length(indS)
#         ii = indS[i]
#         r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
#         r[ii] = 1;
#         Gc[i,:] = 1im/4*hankelh1(0, k*r)*h^2;
#         Gc[i,ii]= 1im/4*D0*h^2;
#     end
#     return Gc
# end

function buildFastConvolution(x::Array{Float64,1},y::Array{Float64,1},
                              h::Float64,k,nu::Function; quadRule::String = "trapezoidal")

  if quadRule == "trapezoidal"

    (ppw,D) = referenceValsTrapRule();
    D0      = D[round(Int,k*h)];
    (n,m) = length(x), length(y)
    Ge    = buildGConv(x,y,h,n,m,D0,k);
    GFFT  = fft(Ge);
    X = repmat(x, 1, m)[:]
    Y = repmat(y', n,1)[:]

    return FastM(GFFT,nu(X,Y),2*n-1,2*m-1,n, m, k);

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
        return FastM(GFFT, nu(X,Y), 4*n, 4*m,
                     n, m, k , quadRule="Greengard_Vico");
      else
        # kx = (-2*(n-1):1:2*(n-1) )/4;
        # ky = (-2*(m-1):1:2*(m-1) )/4;

        # KX = (2*pi/Lp)*repmat(kx, 1, 4*m-3);
        # KY = (2*pi/Lp)*repmat(ky', 4*n-3,1);

        # S = sqrt(KX.^2 + KY.^2);

        # GFFT = Gtruncated2D(L, k, S)

        # return FastM(GFFT,nu(X,Y),4*n-3,4*m-3,
        #              n,m, k,quadRule = "Greengard_Vico");

        kx = (-2*n:1:2*n-1);
        ky = (-2*m:1:2*m-1);

        KX = (2*pi/Lp)*repmat( kx, 1,4*m);
        KY = (2*pi/Lp)*repmat(ky',4*n,  1);

        S = sqrt(KX.^2 + KY.^2);

        GFFT = Gtruncated2D(L, k, S)

        return FastM(GFFT,nu(X,Y),4*n,4*m,
                     n,m, k,quadRule = "Greengard_Vico");


    end
  end
end



# we need to write the convolution in 3D, the aim is to have 2 and 3 convolution
function buildFastConvolution3D(x,y,z,X,Y,Z,h,k,nu; quadRule::String = "Greengard_Vico")

 if quadRule == "Greengard_Vico"

      Lp = 4*(abs(x[end] - x[1]) + h)
      L = (abs(x[end] - x[1]) + h)*1.8
      (n,m,l) = length(x), length(y), length(z)
      #LPhysical = abs(x[end]-x[1])+h;

      if mod(n,2) == 0

        kx = (2*pi/Lp)*collect(-(2*n):1:(2*n-1));
        ky = (2*pi/Lp)*collect(-(2*m):1:(2*m-1));
        kz = (2*pi/Lp)*collect(-(2*l):1:(2*l-1));

        # KX = 2*pi*[ kx[i] + 0*j   + 0*p   for i=1:4*n, j=1:4*m, p=1:4*l ]
        # KY = 2*pi*[ 0*i   + ky[j] + 0*p   for i=1:4*n, j=1:4*m, p=1:4*l ]
        # KZ = 2*pi*[ 0*i   + 0*j   + kz[p] for i=1:4*n, j=1:4*m, p=1:4*l ]

        # S = sqrt(KX.^2 + KY.^2 + KZ.^2);
        ## S  = [ sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2) for i=1:4*n, j=1:4*m, p=1:4*l ]
        
        # GFFT = Gtruncated3D(L, k, S)

        ### GFFT = [ Gtruncated3D(L,k,sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2)) for i=1:4*n, j=1:4*m, p=1:4*l]

        # Computing the convolution kernel (we just use a for loop in order to save memory)
        GFFT = zeros(Complex128, 4*n, 4*m, 4*l)

        for ii=1:4*n, jj=1:4*m, pp=1:4*l
          GFFT[ii,jj,pp]  = Gtruncated3D(L,k,sqrt(kx[ii]^2 + ky[jj]^2 + kz[pp]^2));
        end

        return FastM3D(GFFT,nu(X,Y,Z),4*n,4*m,4*l, n, m,l, k ,quadRule = "Greengard_Vico");
      else

        kx = 2*pi*collect(-2*(n-1):1:2*(n-1) )/4;
        ky = 2*pi*collect(-2*(m-1):1:2*(m-1) )/4;
        kz = 2*pi*collect(-2*(m-1):1:2*(m-1) )/4;

        # KX = [ kx[i] + 0*j   + 0*p   for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]
        # KY = [ 0*i   + ky[j] + 0*p   for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]
        # KZ = [ 0*i   + 0*j   + kz[p] for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]

        # S = sqrt(KX.^2 + KY.^2 + KZ.^2);

        ## S  = [ sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2) for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]

        # GFFT = Gtruncated3D(L, k, S)

        ### GFFT = [ Gtruncated3D(L,k,sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2)) for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]

        # Computing the convolution kernel (we just use a for loop in order to save memory)
        GFFT = zeros(Complex128, 4*n-3, 4*m-3, 4*l-3)

        for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3
          GFFT[i,j,p]  = Gtruncated3D(L,k,sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2));
        end

        return FastM3D(GFFT,nu(X,Y,Z),4*n-3,4*m-3,4*l-3,n, m, l,k,quadRule = "Greengard_Vico");

    end
  end
end



@everywhere function sampleG(k,X,Y,indS, D0)
    # function to sample the Green's function at frequency k

  #   R  = SharedArray(Float64, length(indS), length(X))
  #   Xshared = SharedArray(X)
  #   Yshared = SharedArray(Y)
  #   @sync begin
  #     @parallel for i = 1:length(indS)
  #   #for i = 1:length(indS)
  #     ii = indS[i]
  #     R[i,:]  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
  #     R[i,ii] = 1;
  #   end
  # end

  R  = SharedArray(Float64, length(indS), length(X))
  Xshared = convert(SharedArray, X)
  Yshared = convert(SharedArray, Y)
  @sync begin
    @parallel for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,:]  = sqrt( (Xshared-Xshared[ii]).^2 + (Yshared-Yshared[ii]).^2);
      R[i,ii] = 1;
    end
  end

    #Gc = (h^2*1im/4)*hankelh1(0, k*R)*h^2;
    Gc = sampleGkernelpar(k,R,h);
    for i = 1:length(indS)
        ii = indS[i]
        Gc[i,ii]= 1im/4*D0*h^2;
    end
    return Gc
end

@everywhere function sampleG3D(k,X,Y,Z, indS, D0)
  # function to sample the Green's function at frequency k

  R  = SharedArray(Float64, length(indS), length(X))
  Xshared = convert(SharedArray, X)
  Yshared = convert(SharedArray, Y)
  Zshared = convert(SharedArray, Z)
  @sync begin
    @parallel for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,:]  = sqrt( (Xshared-Xshared[ii]).^2 + (Yshared-Yshared[ii]).^2 + (Zshared-Zshared[ii]).^2);
      R[i,ii] = 1;
    end
  end

    Gc = (exp(1im*k*R)*h^2)./(4*pi*R) ;
    for i = 1:length(indS)
        ii = indS[i]
        Gc[i,ii]= 1im/4*D0*h^2;
    end
    return Gc
end

## TODO: parallelize the sampling of the 3D Green's function
# @everywhere function sampleG3DParallel(k,X,Y,Z, indS, D0)
#   # function to sample the Green's function at frequency k

#   R  = SharedArray(Float64, length(indS), length(X))
#   Xshared = convert(SharedArray, X)
#   Yshared = convert(SharedArray, Y)
#   Zshared = convert(SharedArray, Z)
#   @sync begin
#       for i = 1:length(indS)
#       #for i = 1:length(indS)
#       ii = indS[i]
#       R[i,:]  = sqrt( (Xshared-Xshared[ii]).^2 + (Yshared-Yshared[ii]).^2 + (Zshared-Zshared[ii]).^2);
#       R[i,ii] = 1;
#     end
#   end

#     Gc = (exp(1im*k*R)*h^2)./(4*pi*R) ;
#     for i = 1:length(indS)
#         ii = indS[i]
#         Gc[i,ii]= 1im/4*D0*h^2;
#     end
#     return Gc
# end



@everywhere function myrange(q::SharedArray)
    idx = indexpids(q)
    if idx == 0
        # This worker is not assigned a piece
        return 1:0, 1:0
    end
    nchunks = length(procs(q))
    splits = [round(Int, s) for s in linspace(0,size(q,2),nchunks+1)]
    1:size(q,1), splits[idx]+1:splits[idx+1]
end

@everywhere function sampleGkernelpar(k,r::Array{Float64,1},h)
  n  = length(r)
  println("Sample kernel parallel loop ")
  G = SharedArray(Complex128,n)
  rshared = convert(SharedArray, r)
  @sync @parallel for ii = 1:n
          @inbounds  G[ii] = 1im/4*hankelh1(0, k*rshared[ii])*h^2;
  end
  return sdata(G)
end

## two different versions of the same function with slight different input

@everywhere function sampleGkernelpar(k,R::Array{Float64,2},h)
  (m,n)  = size(R)
  println("Sample kernel parallel loop with chunks ")
  G = SharedArray(Complex128,m,n)
  @time Rshared = convert(SharedArray, R)
  @sync begin
        for p in procs(G)
             @async remotecall_fetch(sampleGkernel_shared_chunk!,p, G, Rshared,k,h)
        end
    end
  return sdata(G)
end

@everywhere function sampleGkernelpar(k,R::SharedArray{Float64,2},h)
  (m,n)  = size(R)
  #println("Sample kernel parallel loop with chunks 2 ")
  G = SharedArray(Complex128,m,n)
  @sync begin
        for p in procs(G)
            @async remotecall_fetch(sampleGkernel_shared_chunk!,p, G, R,k,h)
        end
    end
  return sdata(G)
end


# little convenience wrapper
@everywhere sampleGkernel_shared_chunk!(q,u,k,h) = sampleGkernel_chunk!(q,u,k,h, myrange(q)...)

@everywhere @inline function sampleGkernel_chunk!(G, R,k::Float64,h::Float64, 
                                                  irange::UnitRange{Int64}, jrange::UnitRange{Int64})
    #@show (irange, jrange)  # display so we can see what's happening
    # println(myid())
    # println(typeof(irange))
    alpha = 1im/4*h^2
    for i in irange
      for j in jrange 
        @inbounds G[i,j] = alpha*hankelh1(0, k*R[i,j]);
      end
    end
end

function entriesSparseA(k,X,Y,D0, n ,m)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex128}[]
  Indices  = Array{Int64}[]

  IndRelative = zeros(Int64,3,3)
  IndRelative = [-n-1 -n -n+1;
                   -1  0    1;
                  n-1  n  n+1]

  N = n*m;

  # computing the entries for the interior
  # extracting indices at the center of the stencil
  indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + IndRelative[:] );
  # extracting only the far field indices 
  indVolC = setdiff(collect(1:N),indVol);
  GSampled = sampleG(k,X,Y,indVol, D0)[:,indVolC];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[:]);

  # this is for the edges 

  # for  x = xmin, y = 0
  indFz1 = round(Integer, n*(m-1)/2 +1 + IndRelative[:,2:3][:]);
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG(k,X,Y,indFz1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[:,2:3][:]); #'

  # for  x = xmax, y = 0
  indFz2 = round(Integer, n*(m-1)/2 + IndRelative[:,1:2][:]);
  indC = setdiff(collect(1:N),indFz2);
  GSampled = sampleG(k,X,Y,indFz2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[:,1:2][:]); #'

  # for  y = ymin, x = 0
  indFx1 = round(Integer, (n+1)/2 + IndRelative[2:3,:][:]);
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG(k,X,Y,indFx1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[2:3,:][:]); #'

  # for  y = ymin, x = 0
  indFx2 = round(Integer, N - (n+1)/2 + IndRelative[1:2,:][:]);
  indC = setdiff(collect(1:N),indFx2);
  GSampled = sampleG(k,X,Y,indFx2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[1:2,:][:]); #'

  # For the corners
  indcorner1 = round(Integer, 1       + IndRelative[2:3,2:3][:]);
  indcorner2 = round(Integer, n       + IndRelative[2:3,1:2][:]);
  indcorner3 = round(Integer, n*m-n+1 + [0, 1,-n,-n+1]);
  indcorner4 = round(Integer, n*m     + [0,-1,-n,-n-1]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries, U[:,end]'); #'
  push!(Indices, IndRelative[2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner3);
  GSampled = sampleG(k,X,Y,indcorner3, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,1, -n,-n+1]); #'

  #'
  indC = setdiff(collect(1:N),indcorner4);
  GSampled = sampleG(k,X,Y,indcorner4, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,-1, -n,-n-1]); #'

  return (Indices, Entries)
end


function entriesSparseA3D(k,X,Y,Z,D0, n ,m, l)
  # in this case we need to build everythig with ranodmized methods 
  # we need to have an odd number of points
  #@assert mod(length(X),2) == 1
  Entries  = Array{Complex128}[]
  Indices  = Array{Int64}[]

  N = n*m*l;

  Ind_relative = zeros(Int64,3,3,3)
  Ind_relative[:,:,1] = [(-m*n-n-1) (-m*n-n) (-m*n-n+1);
                         (-m*n  -1) (-m*n  ) (-m*n  +1);
                         (-m*n+n-1) (-m*n+n) (-m*n+n+1) ]';

  Ind_relative[:,:,2] = [-n-1 -n -n+1; 
                           -1  0    1;
                          n-1  n  n+1]'; 

  Ind_relative[:,:,3] = [(m*n-n-1) (m*n-n) (m*n-n+1);
                         (m*n  -1) (m*n  ) (m*n  +1);
                         (m*n+n-1) (m*n+n) (m*n+n+1) ]';                        

  # computing the entries for the interior

  nHalf = round(Integer,n/2);
  mHalf = round(Integer,m/2);
  lHalf = round(Integer,l/2);

  indVol = round(Integer, changeInd3D(nHalf,mHalf,lHalf,n,m,l)+Ind_relative[:]);
  
  indVolC = setdiff(collect(1:N),indVol);
  GSampled = sampleG3D(k,X,Y,Z,indVol, D0)[:,indVolC ];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative);
  

  # for  x = xmin,  y = anything z = anything
  indFx1 = round(Integer, changeInd3D(1,mHalf,lHalf,n,m,l) + Ind_relative[2:3,:,:][:] );
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG3D(k,X,Y,Z,indFx1, D0)[:,indC ];
   # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,:,:][:] );#'

  
  # for  x = xmax, y = any z = any
  indFxN = round(Integer, changeInd3D(n,mHalf,lHalf,n,m,l) + Ind_relative[1:2,:,:][:]);
  indC = setdiff(collect(1:N),indFxN);
  GSampled = sampleG3D(k,X,Y,Z,indFxN, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices, Ind_relative[1:2,:,:][:]); #'


  # for  y = ymin, x = any z = any
  indFy1 = round(Integer, changeInd3D(nHalf,1,lHalf,n,m,l) + Ind_relative[:,2:3,:][:] );
  indC = setdiff(collect(1:N),indFy1);
  GSampled = sampleG3D(k,X,Y,Z,indFy1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,2:3,:][:]); #'

  # for  y = ymax, x = any z = any
  indFyN = round(Integer, changeInd3D(nHalf,m,lHalf,n,m,l) + Ind_relative[:,1:2,:][:] );
  indC = setdiff(collect(1:N),indFyN);
  GSampled = sampleG3D(k,X,Y,Z,indFyN, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,1:2,:][:] ); #'

  # for  z = zmin, x = any y = any
  indFz1 = round(Integer, changeInd3D(nHalf,mHalf,1,n,m,l) + Ind_relative[:,:,2:3][:] );
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG3D(k,X,Y,Z,indFz1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,:,2:3][:] ); #'

  # for  z = zmax, x = any y = any
  indFzN = round(Integer, changeInd3D(nHalf,mHalf,l,n,m,l) + Ind_relative[:,:,1:2][:] );
  indC = setdiff(collect(1:N),indFzN);
  GSampled = sampleG3D(k,X,Y,Z,indFzN, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,:,1:2][:] ); #'

  # we need to incorporate the vertices
  indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
  indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
  indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
  indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
  indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(1,2,1,Ind_relative));
  indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(3,2,1,Ind_relative));
  indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(1,2,3,Ind_relative));
  indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(3,2,3,Ind_relative));
  indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,1,1,Ind_relative));
  indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,3,1,Ind_relative));
  indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,1,3,Ind_relative));
  indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,3,3,Ind_relative));


  indC = setdiff(collect(1:N),indvertex1);
  GSampled = sampleG3D(k,X,Y,Z,indvertex1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,1,2,Ind_relative)); #'
  
  indC = setdiff(collect(1:N),indvertex2);
  GSampled = sampleG3D(k,X,Y,Z,indvertex2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,1,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex3);
  GSampled = sampleG3D(k,X,Y,Z,indvertex3, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,3,2,Ind_relative)); #'
  
  indC = setdiff(collect(1:N),indvertex4);
  GSampled = sampleG3D(k,X,Y,Z,indvertex4, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,3,2,Ind_relative)); #'
  
  indC = setdiff(collect(1:N),indvertex5);
  println(indvertex5)
  GSampled = sampleG3D(k,X,Y,Z,indvertex5, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex6);
  GSampled = sampleG3D(k,X,Y,Z,indvertex6, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex7);
  GSampled = sampleG3D(k,X,Y,Z,indvertex7, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex8);
  GSampled = sampleG3D(k,X,Y,Z,indvertex8, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex9);
  GSampled = sampleG3D(k,X,Y,Z,indvertex9, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,1,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex10);
  GSampled = sampleG3D(k,X,Y,Z,indvertex10, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex11);
  GSampled = sampleG3D(k,X,Y,Z,indvertex11, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,1,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex12);
  GSampled = sampleG3D(k,X,Y,Z,indvertex12, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,3,Ind_relative)); #'


  # Now we incorporate the corner
  indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
  indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
  indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
  indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
  indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
  indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
  indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
  indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleG3D(k,X,Y,Z,indcorner1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG3D(k,X,Y,Z,indcorner2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner3);
  GSampled = sampleG3D(k,X,Y,Z,indcorner3, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner4);
  GSampled = sampleG3D(k,X,Y,Z,indcorner4, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,2:3][:]); #'

  indC = setdiff(collect(1:N),indcorner5);
  GSampled = sampleG3D(k,X,Y,Z,indcorner5, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner6);
  GSampled = sampleG3D(k,X,Y,Z,indcorner6, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner7);
  GSampled = sampleG3D(k,X,Y,Z,indcorner7, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner8);
  GSampled = sampleG3D(k,X,Y,Z,indcorner8, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,1:2][:]); #'

  return (Indices, Entries)
end

function referenceValsTrapRule()
    # modification to the Trapezoidal rule for logarithmic singularity
    # as explained in
    # R. Duan and V. Rokhlin, High-order quadratures for the solution of scattering problems in
    # two dimensions, J. Comput. Phys.,
    x = 2.0.^(-(0:5))[:]
    w = [1-0.892*im, 1-1.35*im, 1-1.79*im, 1- 2.23*im, 1-2.67*im, 1-3.11*im]
    return (x,w)
end

@inline function changeInd3D(indI::Int64, indJ::Int64, indK::Int64, n::Int64,m::Int64,k::Int64)
  # function easily compute the global index from dimensional ones
  # for a third order tensor
  return (indK-1)*n*m + (indJ-1)*n + indI
end

@inline function changeInd3D(indI::Array{Int64,1}, indJ::Array{Int64,1}, indK::Array{Int64,1}, n::Int64,m::Int64,k::Int64)
  # function easily compute the global index from dimensional ones
  # for a third order tensor
  return (indK-1)*n*m + (indJ-1)*n + indI
end

@inline function subStencil3D(indI::Int64, indJ::Int64, indK::Int64, Ind_relative::Array{Int64,3})
  # function to easily the local substencil for a given set of indices!
  # for a third order tensor
  ii = max(indI-1,1):min(indI+1,3)
  jj = max(indJ-1,1):min(indJ+1,3)
  kk = max(indK-1,1):min(indK+1,3)

  return Ind_relative[ii,jj,kk][:]
end




# @inline function indexMatrix(indI::Int64, indJ::Int64, indK::Int64, n::Int64,m::Int64,k::Int64)
#   # function easily compute the global index from dimensional ones
#   # for a third order tensor



# end


function buildGConv(x,y,h::Float64,n::Int64,m::Int64,D0,k::Float64)
    # function to build the convolution vector for the
    # fast application of the convolution

    # this is built for odd n and odd m.

    if mod(n,2) == 1
      # build extended domain
      xe = collect((x[1]-(n-1)/2*h):h:(x[end]+(n-1)/2*h));
      ye = collect((y[1]-(m-1)/2*h):h:(y[end]+(m-1)/2*h));

      Xe = repmat(xe, 1, 2*m-1);
      Ye = repmat(ye', 2*n-1,1);


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

    # to avoid evaluating at the singularity
    indMiddle = find(R.==0)[1]    # we modify R to remove the zero (so we don't )
    R[indMiddle] = 1;
    # sampling the Green's function
    Ge = sampleGkernelpar(k,R,h)
    #Ge = pmap( x->1im/4*hankelh1(0,k*x)*h^2, R)
    # modiyfin the diagonal with the quadrature
    # modification
    Ge[indMiddle] = 1im/4*D0*h^2;

    return Ge

end


function buildGConvPar(x,y,h,n,m,D0,k)

    # build extended domain
    xe = collect((x[1]-(n-1)/2*h):h:(x[end]+(n-1)/2*h));
    ye = collect((y[1]-(m-1)/2*h):h:(y[end]+(m-1)/2*h));

    Xe = repmat(xe, 1, 2*m-1);
    Ye = repmat(ye', 2*n-1,1);

    R = sqrt(Xe.^2 + Ye.^2);
    # to avoid evaluating at the singularity
    indMiddle = round(Integer, m)
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



function createIndices(row::Array{Int64,1}, col::Array{Int64,1}, val::Array{Complex128,1})
  # function to create the indices for a sparse matrix
  @assert length(col) == length(val)
  nn = length(col);
  mm = length(row);

  Row = kron(row, ones(Int64, nn));
  Col = kron(ones(Int64,mm), col) + Row;
  Val = kron(ones(Int64,mm), val)
  return (Row,Col,Val)
end

function createIndices(row::Int64, col::Array{Int64,1}, val::Array{Complex128,1})

  @assert length(col) == length(val)
  nn = length(col);
  mm = 1;

  Row = kron(row, ones(Int64, nn));
  Col = kron(ones(Int64,mm), col) + Row;
  Val = kron(ones(Int64,mm), val)
  return (Row,Col,Val)
end

function buildConvMatrix(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},D0::Complex128, h::Float64)
    # function to build the convolution matrix
    @assert length(X) == length(Y)
    N = length(X);

    G = zeros(Complex128, N, N);

    r = zeros(Float64,N)
    for ii = 1:N
            r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
            r[ii] = 1;
            G[ii,:] =  1im/4*hankelh1(0, k*r)*h^2;
            G[ii,ii]=  1im/4*D0*h^2;
    end

    return G
end




function buildSparseA(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       D0::Complex128, n::Int64 ,m::Int64; method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    if method == "normal"
      (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);
    elseif method == "randomized"
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end


    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1][:],
                                    Indices[1][:], Values[1][:]);

    (Row, Col, Val) = createIndices(Ind[1,2:end-1][:],
                                    Indices[2][:], Values[2][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);


    (Row, Col, Val) = createIndices(Ind[end,2:end-1][:],
                                    Indices[3][:], Values[3][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[2:end-1,1][:],
                                    Indices[4][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[2:end-1,end][:],
                                    Indices[5][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[1,1],
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[end,1],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[1,end],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[end,end],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    A = sparse(rowA,colA,valA);

    return A;
end


## To be done! (I don't remember how I built this one)
function buildSparseA3D(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                        Z::Array{Float64,1},
                        D0::Complex128, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    # building the indices, columns and rows for the interior
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], Values[1][:]);

   
    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], Values[2][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], Values[3][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[1,1],
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[end,1],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[1,end],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[end,end],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    A = sparse(rowA,colA,valA);

    return A;
end

function entriesSparseG(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       D0::Complex128, n::Int64 ,m::Int64)
  # function to compute the entried of G, inside the volume, at the boundaries
  # and at the corners. This allows us to compute A*G in O(n) time instead of
  # O(n^2)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex128}[]

  IndRelative = zeros(Int64,3,3)
  IndRelative = [-n-1 -n -n+1;
                   -1  0    1;
                  n-1  n  n+1]

  N = n*m;

  # computing the entries for the interior
  indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + IndRelative[:] );
  GSampled = sampleG(k,X,Y,indVol, D0)[:,indVol];

  push!(Entries,GSampled);

  # for  x = xmin, y = 0
  indFz1 = round(Integer, n*(m-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
  GSampled = sampleG(k,X,Y,indFz1, D0)[:,indFz1];

  push!(Entries,GSampled);

  # for  x = xmax, y = 0
  indFz2 = round(Integer, n*(m-1)/2 + [-1,0,n,n-1,-n, -n-1]);
  GSampled = sampleG(k,X,Y,indFz2, D0)[:,indFz2];

  push!(Entries,GSampled);

  # for  y = ymin, x = 0
  indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
  GSampled = sampleG(k,X,Y,indFx1, D0)[:,indFx1];

  push!(Entries,GSampled);


  # for  y = ymin, x = 0
  indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
  GSampled = sampleG(k,X,Y,indFx2, D0)[:,indFx2];

  push!(Entries,GSampled);

  # For the corners
  indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
  indcorner2 = round(Integer, n + [0,-1, n,n-1]);
  indcorner3 = round(Integer, n*m-n+1 + [0,1, -n,-n+1]);
  indcorner4 = round(Integer, n*m + [0,-1, -n,-n-1]);

  GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indcorner1];
  push!(Entries,GSampled);


  #'
  GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indcorner2];
  push!(Entries,GSampled);

  #'
  GSampled = sampleG(k,X,Y,indcorner3, D0)[:,indcorner3];
  push!(Entries,GSampled);

  #'
  GSampled = sampleG(k,X,Y,indcorner4, D0)[:,indcorner4];
  push!(Entries,GSampled);

  return Entries
end


function buildSparseAG(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       D0::Complex128, n::Int64 ,m::Int64; method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    if method=="normal"
      (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);
    elseif method == "randomized" 
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    Entries = entriesSparseG(k,X,Y,D0, n ,m);

    ValuesAG = Values[1]*Entries[1];
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1][:],
                                    Indices[1][:], ValuesAG[:]);

    ValuesAG = Values[2]*Entries[2];
    (Row, Col, Val) = createIndices(Ind[1,2:end-1][:],
                                    Indices[2][:], ValuesAG[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    ValuesAG = Values[3]*Entries[3];
    (Row, Col, Val) = createIndices(Ind[end,2:end-1][:],
                                    Indices[3][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[4]*Entries[4];
    (Row, Col, Val) = createIndices(Ind[2:end-1,1][:],
                                    Indices[4][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[5]*Entries[5];
    (Row, Col, Val) = createIndices(Ind[2:end-1,end][:],
                                    Indices[5][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[6]*Entries[6];
    (Row, Col, Val) = createIndices(Ind[1,1],
                                    Indices[6][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[7]*Entries[7];
    (Row, Col, Val) = createIndices(Ind[end,1],
                                    Indices[7][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[8]*Entries[8];
    (Row, Col, Val) = createIndices(Ind[1,end],
                                    Indices[8][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[9]*Entries[9];
    (Row, Col, Val) = createIndices(Ind[end,end],
                                    Indices[9][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    AG = sparse(rowA,colA,valA);

    return AG;
end

######################## Randomized methods ##############################
######not working 

# function entriesSparseARand(k,X,Y,D0, n ,m)
#   # we need to have an even number of points
#   # We compute the entries for the matrix A using randomized methods
#   @assert mod(length(X),2) == 1
#   Entries  = Array{Complex128}[]
#   Indices  = Array{Int64}[]

#   N = n*m;

#   # computing the entries for the interior
#   indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);
#   indVolC = setdiff(collect(1:N),indVol);
#   GSampled = sampleG(k,X,Y,indVol, D0)[:,indVolC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);

#   # for  x = xmin, y = 0
#   indFz1 = round(Integer, n*(m-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
#   indC = setdiff(collect(1:N),indFz1);
#   GSampled = sampleG(k,X,Y,indFz1, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices, [0,1,n,n+1,-n, -n+1]); #'

#   # for  x = xmax, y = 0
#   indFz2 = round(Integer, n*(n-1)/2 + [-1,0,n,n-1,-n, -n-1]);
#   indC = setdiff(collect(1:N),indFz2);
#   GSampled = sampleG(k,X,Y,indFz2, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,n,n-1,-n, -n-1]); #'

#   # for  y = ymin, x = 0
#   indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
#   indC = setdiff(collect(1:N),indFx1);
#   GSampled = sampleG(k,X,Y,indFx1, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,1,n,n+1, n-1]); #'

#   # for  y = ymin, x = 0
#   indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
#   indC = setdiff(collect(1:N),indFx2);
#   GSampled = sampleG(k,X,Y,indFx2, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,1,-n,-n+1, -n-1]); #'

#   # For the corners
#   indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
#   indcorner2 = round(Integer, n + [0,-1, n,n-1]);
#   indcorner3 = round(Integer, n*m-n+1 + [0,1, -n,-n+1]);
#   indcorner4 = round(Integer, n*m + [0,-1, -n,-n-1]);

#   indC = setdiff(collect(1:N),indcorner1);
#   GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indC ];
#   # computing the sparsifying correction
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,1, n,n+1]); #'

#   #'
#   indC = setdiff(collect(1:N),indcorner2);
#   GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indC ];
#   # computing the sparsifying correction
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,-1, n,n-1]); #'

#   #'
#   indC = setdiff(collect(1:N),indcorner3);
#   GSampled = sampleG(k,X,Y,indcorner3, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,1, -n,-n+1]); #'

#   #'
#   indC = setdiff(collect(1:N),indcorner4);
#   GSampled = sampleG(k,X,Y,indcorner4, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,-1, -n,-n-1]); #'

#   return (Indices, Entries)
# end
