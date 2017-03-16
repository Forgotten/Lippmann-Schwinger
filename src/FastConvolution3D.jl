# File with the function necessary to implement the fast convolution
# in 3D and all the necessary machinery to build the preconditioner


type FastM3D
    ## we may want to add an FFT plan to make the evaluation faster
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

import Base.*


function *(M::FastM3D, b::Array{Complex128,1}; verbose::Bool=false)
    # multiply by nu, compute the convolution and then
    # multiply by omega^2
    B = M.omega^2*(FFTconvolution(M,M.nu.*b, verbose=verbose))

    return (b + B)
end

@inline function FFTconvolution(M::FastM3D, b::Array{Complex128,1};
                                verbose::Bool=false )
    # function to compute the convolution with the convolution kernel
    # defined within the FastM3D type using the FFT
    # TODO add a fft plan in here to accelerate the speed
    # input: M::FastM3D type containing the convolution kernel
    #        b::Array{Complex128,1} vector to apply the conv kernel
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


@everywhere function sampleG3D(k,X,Y,Z, indS, fastconv::FastM3D)
  # function to sample the 3D Green's function in the nodes given by indS
  # input:    k: float64 frequency
  #           X: mesh contaning the X position of each point
  #           Y: mesh contaning the Y position of each point
  #           Z: mesh contaning the Y position of each point
  #           indS: indices in which the sources are located
  #           fastconv: FastM3D type for the application of the
  #                     discrete convolution kernel

  R  = zeros(Complex128, length(indS), length(X))
   for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,ii] = 1;
    end

   Gc =  zeros(Complex128, length(indS), length(X))
   # this can be parallelized but then, we may have cache
   # aceess problem
    for i = 1:length(indS)
        Gc[i,:]= FFTconvolution(fastconv, R[i,:][:])
    end
    return Gc
end

@everywhere function sampleG3D(k,X,Y,Z, indS, D0)
  # function to sample the 3D Green's function in the nodes given by indS
  # input:    k: float64 frequency
  #           X: mesh contaning the X position of each point
  #           Y: mesh contaning the Y position of each point
  #           Z: mesh contaning the Y position of each point
  #           indS: indices in which the sources are located
  #           D0: diagonal modification based on the kh, this is given as a
  #               parameter but it should be correctly handled in newer versions
  #               of the code.
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

  # sampling the Green's function in the given points
    Gc = (exp(1im*k*R)*h^2)./(4*pi*R) ;
    for i = 1:length(indS)
        ii = indS[i]
        Gc[i,ii]= 1im/4*D0*h^2;
    end
    return Gc
end

#######################################################
### function to enhance parallelism
##################################################
# function SampleGtruncated3D(GFFT, L,k,kx,ky,kz)
#     n = length(kx)
#     m = length(ly)
#     l = length(kz)

#     R  = SharedArray(Float64, length(indS), length(X))
#     kxshared = convert(SharedArray, kx)
#     kyshared = convert(SharedArray, ky)
#     kzshared = convert(SharedArray, kz)

# end




# @everywhere function myrange(q::SharedArray)
#     idx = indexpids(q)
#     if idx == 0
#         # This worker is not assigned a piece
#         return 1:0, 1:0
#     end
#     nchunks = length(procs(q))
#     splits = [round(Int, s) for s in linspace(0,size(q,2),nchunks+1)]
#     1:size(q,1), splits[idx]+1:splits[idx+1]
# end

# @everywhere function sampleGkernelparTruncated3D(k,r::Array{Float64,1},h)
#   n  = length(r)
#   println("Sample kernel parallel loop ")
#   G = SharedArray(Complex128,n)
#   rshared = convert(SharedArray, r)
#   @sync @parallel for ii = 1:n
#           @inbounds  G[ii] = 1im/4*hankelh1(0, k*rshared[ii])*h^2;
#   end
#   return sdata(G)
# end

# ## two different versions of the same function with slight different input

# @everywhere function sampleGkernelpar(k,R::Array{Float64,2},h)
#   (m,n)  = size(R)
#   println("Sample kernel parallel loop with chunks ")
#   G = SharedArray(Complex128,m,n)
#   @time Rshared = convert(SharedArray, R)
#   @sync begin
#         for p in procs(G)
#              @async remotecall_fetch(sampleGkernel_shared_chunk!,p, G, Rshared,k,h)
#         end
#     end
#   return sdata(G)
# end


# # little convenience wrapper
# @everywhere sampleGkernel_shared_chunk!(q,u,k,h) = sampleGkernel_chunk!(q,u,k,h, myrange(q)...)

# @everywhere @inline function sampleGkernel_chunk!(G,R,k::Float64,h::Float64,
#                                                   irange::UnitRange{Int64}, jrange::UnitRange{Int64})
#     #@show (irange, jrange)  # display so we can see what's happening
#     # println(myid())
#     # println(typeof(irange))
#     alpha = 1im/4*h^2
#     for i in irange
#       for j in jrange
#         @inbounds G[i,j] = alpha*hankelh1(0, k*R[i,j]);
#       end
#     end
# end


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

        # This loop can be easily parallelized
        for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3
          GFFT[i,j,p]  = Gtruncated3D(L,k,sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2));
        end

        return FastM3D(GFFT,nu(X,Y,Z),4*n-3,4*m-3,4*l-3,n, m, l,k,quadRule = "Greengard_Vico");

    end
  end
end





function buildSparseAG3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner

    # Quick fix: TODO change this!


    Entries = entriesSparseG3D(k,X,Y,Z,fastconv, n ,m, l);
    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    ValuesAG = Values[1]*Entries[1];


    # building the indices, columns and rows for the interior
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], (Values[1]*Entries[1])[ :]);


    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], (Values[2]*Entries[2])[ :]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], (Values[3]*Entries[3])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], (Values[4]*Entries[4])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], (Values[5]*Entries[5])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], (Values[6]*Entries[6])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], (Values[7]*Entries[7])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)





# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], (Values[8]*Entries[8])[ :]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], (Values[9]*Entries[9])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], (Values[10]*Entries[10])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], (Values[11]*Entries[11])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], (Values[12]*Entries[12])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], (Values[13]*Entries[13])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], (Values[14]*Entries[14])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], (Values[15]*Entries[15])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], (Values[16]*Entries[16])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], (Values[17]*Entries[17])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], (Values[18]*Entries[18])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], (Values[19]*Entries[19])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], (Values[20]*Entries[20])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], (Values[21]*Entries[21])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], (Values[22]*Entries[22])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], (Values[23]*Entries[23])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], (Values[24]*Entries[24])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], (Values[25]*Entries[25])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], (Values[26]*Entries[26])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], (Values[27]*Entries[27])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)



    AG = sparse(rowA,colA,valA);

    return AG;
end



@everywhere function sampleG3D(k,X,Y,Z, indS, fastconv::FastM3D)
  # function to sample the 3D Green's function in the nodes given by indS
  # input:    k: float64 frequency
  #           X: mesh contaning the X position of each point
  #           Y: mesh contaning the Y position of each point
  #           Z: mesh contaning the Y position of each point
  #           indS: indices in which the sources are located
  #           D0: how to compute the qudrature fast
  # function to sample the Green's function at frequency k

  R  = zeros(Complex128, length(indS), length(X))
   for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,ii] = 1;
    end

   Gc =  zeros(Complex128, length(indS), length(X))
    for i = 1:length(indS)
        Gc[i,:]= FFTconvolution(fastconv, R[i,:][:])
    end
    return Gc
end

function entriesSparseG3D(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64)
  # function to compute the entried of G, inside the volume, at the boundaries
  # and at the corners. This allows us to compute A*G in O(n) time instead of
  # O(n^2)
  # we need to have an even number of points
  #

  Entries  = Array{Complex128}[]

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

  GSampled = sampleG3D(k,X,Y,Z,indVol, fastconv)[:,indVol];

  push!(Entries,GSampled);


  # for  x = xmin,  y = anything z = anything
  indFx1 = round(Integer, changeInd3D(1,mHalf,lHalf,n,m,l) + Ind_relative[2:3,:,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFx1, fastconv)[:,indFx1];

  push!(Entries,GSampled);


  # for  x = xmax, y = any z = any
  indFxN = round(Integer, changeInd3D(n,mHalf,lHalf,n,m,l) + Ind_relative[1:2,:,:][:]);
  GSampled = sampleG3D(k,X,Y,Z,indFxN, fastconv)[:,indFxN];

  push!(Entries,GSampled);

  # for  y = ymin, x = any z = any
  indFy1 = round(Integer, changeInd3D(nHalf,1,lHalf,n,m,l) + Ind_relative[:,2:3,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFy1, fastconv)[:,indFy1];

  push!(Entries,GSampled);

  # for  y = ymax, x = any z = any
  indFyN = round(Integer, changeInd3D(nHalf,m,lHalf,n,m,l) + Ind_relative[:,1:2,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFyN, fastconv)[:,indFyN];

  push!(Entries,GSampled);

  # for  z = zmin, x = any y = any
  indFz1 = round(Integer, changeInd3D(nHalf,mHalf,1,n,m,l) + Ind_relative[:,:,2:3][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFz1, fastconv)[:,indFz1];

  push!(Entries,GSampled);

  # for  z = zmax, x = any y = any
  indFzN = round(Integer, changeInd3D(nHalf,mHalf,l,n,m,l) + Ind_relative[:,:,1:2][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFzN, fastconv)[:,indFzN];

  push!(Entries,GSampled);

  # we need to incorporate the vertices
  indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
  indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
  indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
  indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
  indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
  indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
  indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
  indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
  indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
  indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
  indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
  indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));

  GSampled = sampleG3D(k,X,Y,Z,indvertex1, fastconv)[:,indvertex1 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex2, fastconv)[:,indvertex2 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex3, fastconv)[:,indvertex3 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex4, fastconv)[:,indvertex4 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex5, fastconv)[:,indvertex5 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex6, fastconv)[:,indvertex6 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex7, fastconv)[:,indvertex7 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex8, fastconv)[:,indvertex8 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex9, fastconv)[:,indvertex9 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex10, fastconv)[:,indvertex10 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex11, fastconv)[:,indvertex11 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex12, fastconv)[:,indvertex12 ];
  push!(Entries,GSampled);


  # Now we incorporate the corners
  indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
  indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
  indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
  indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
  indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
  indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
  indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
  indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);

  GSampled = sampleG3D(k,X,Y,Z,indcorner1, fastconv)[:,indcorner1 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner2, fastconv)[:,indcorner2 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner3, fastconv)[:,indcorner3 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner4, fastconv)[:,indcorner4 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner5, fastconv)[:,indcorner5 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indcorner6, fastconv)[:,indcorner6 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner7, fastconv)[:,indcorner7 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner8, fastconv)[:,indcorner8 ];
  push!(Entries,GSampled);

  return Entries

end




function entriesSparseA3D(k,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},fastconv::FastM3D,
                          n::Int64 ,m::Int64, l::Int64)
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
  GSampled = sampleG3D(k,X,Y,Z,indVol, fastconv)[:,indVolC ];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative);


  # for  x = xmin,  y = anything z = anything
  indFx1 = round(Integer, changeInd3D(1,mHalf,lHalf,n,m,l) + Ind_relative[2:3,:,:][:] );
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG3D(k,X,Y,Z,indFx1, fastconv)[:,indC ];
   # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,:,:][:] );#'


  # for  x = xmax, y = any z = any
  indFxN = round(Integer, changeInd3D(n,mHalf,lHalf,n,m,l) + Ind_relative[1:2,:,:][:]);
  indC = setdiff(collect(1:N),indFxN);
  GSampled = sampleG3D(k,X,Y,Z,indFxN, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices, Ind_relative[1:2,:,:][:]); #'


  # for  y = ymin, x = any z = any
  indFy1 = round(Integer, changeInd3D(nHalf,1,lHalf,n,m,l) + Ind_relative[:,2:3,:][:] );
  indC = setdiff(collect(1:N),indFy1);
  GSampled = sampleG3D(k,X,Y,Z,indFy1, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,2:3,:][:]); #'

  # for  y = ymax, x = any z = any
  indFyN = round(Integer, changeInd3D(nHalf,m,lHalf,n,m,l) + Ind_relative[:,1:2,:][:] );
  indC = setdiff(collect(1:N),indFyN);
  GSampled = sampleG3D(k,X,Y,Z,indFyN, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,1:2,:][:] ); #'

  # for  z = zmin, x = any y = any
  indFz1 = round(Integer, changeInd3D(nHalf,mHalf,1,n,m,l) + Ind_relative[:,:,2:3][:] );
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG3D(k,X,Y,Z,indFz1, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,:,2:3][:] ); #'

  # for  z = zmax, x = any y = any
  indFzN = round(Integer, changeInd3D(nHalf,mHalf,l,n,m,l) + Ind_relative[:,:,1:2][:] );
  indC = setdiff(collect(1:N),indFzN);
  GSampled = sampleG3D(k,X,Y,Z,indFzN, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,:,1:2][:] ); #'

  # we need to incorporate the vertices
  indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
  indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
  indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
  indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
  indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
  indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
  indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
  indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
  indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
  indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
  indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
  indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));


  indC = setdiff(collect(1:N),indvertex1);
  GSampled = sampleG3D(k,X,Y,Z,indvertex1, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,3,2,Ind_relative)); #'


  indC = setdiff(collect(1:N),indvertex2);
  GSampled = sampleG3D(k,X,Y,Z,indvertex2, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,3,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex3);
  GSampled = sampleG3D(k,X,Y,Z,indvertex3, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,1,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex4);
  GSampled = sampleG3D(k,X,Y,Z,indvertex4, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,1,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex5);
  GSampled = sampleG3D(k,X,Y,Z,indvertex5, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex6);
  GSampled = sampleG3D(k,X,Y,Z,indvertex6, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex7);
  GSampled = sampleG3D(k,X,Y,Z,indvertex7, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex8);
  GSampled = sampleG3D(k,X,Y,Z,indvertex8, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex9);
  GSampled = sampleG3D(k,X,Y,Z,indvertex9, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex10);
  GSampled = sampleG3D(k,X,Y,Z,indvertex10, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,1,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex11);
  GSampled = sampleG3D(k,X,Y,Z,indvertex11, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex12);
  GSampled = sampleG3D(k,X,Y,Z,indvertex12, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,1,1,Ind_relative)); #'


  # Now we incorporate the corners
  indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
  indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
  indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
  indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
  indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
  indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
  indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
  indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleG3D(k,X,Y,Z,indcorner1, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG3D(k,X,Y,Z,indcorner2, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner3);
  GSampled = sampleG3D(k,X,Y,Z,indcorner3, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner4);
  GSampled = sampleG3D(k,X,Y,Z,indcorner4, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,2:3][:]); #'

  indC = setdiff(collect(1:N),indcorner5);
  GSampled = sampleG3D(k,X,Y,Z,indcorner5, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner6);
  GSampled = sampleG3D(k,X,Y,Z,indcorner6, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner7);
  GSampled = sampleG3D(k,X,Y,Z,indcorner7, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner8);
  GSampled = sampleG3D(k,X,Y,Z,indcorner8, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,1:2][:]); #'

  return (Indices, Entries)
end


## To be done! (I don't remember how I built this one)
function buildSparseA3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                        Z::Array{Float64,1},
                        fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner
# this varians uses the kernel from the discretization of the integral system


    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,fastconv, n ,m);
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
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], Values[10][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], Values[11][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], Values[12][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], Values[13][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], Values[14][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], Values[15][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], Values[16][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], Values[17][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], Values[18][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], Values[19][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], Values[20][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], Values[21][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], Values[22][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], Values[23][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], Values[24][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], Values[25][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], Values[26][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], Values[27][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    A = sparse(rowA,colA,valA);

    return A;
end




function buildSparseAG3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner

    # Quick fix: TODO change this!


    Entries = entriesSparseG3D(k,X,Y,Z,fastconv, n ,m, l);
    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    ValuesAG = Values[1]*Entries[1];


    # building the indices, columns and rows for the interior
    (rowAG, colAG, valAG) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], (Values[1]*Entries[1])[ :]);


    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], (Values[2]*Entries[2])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], (Values[3]*Entries[3])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], (Values[4]*Entries[4])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], (Values[5]*Entries[5])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], (Values[6]*Entries[6])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], (Values[7]*Entries[7])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)





# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], (Values[8]*Entries[8])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], (Values[9]*Entries[9])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], (Values[10]*Entries[10])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], (Values[11]*Entries[11])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], (Values[12]*Entries[12])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], (Values[13]*Entries[13])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], (Values[14]*Entries[14])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], (Values[15]*Entries[15])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], (Values[16]*Entries[16])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], (Values[17]*Entries[17])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], (Values[18]*Entries[18])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], (Values[19]*Entries[19])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], (Values[20]*Entries[20])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], (Values[21]*Entries[21])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], (Values[22]*Entries[22])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], (Values[23]*Entries[23])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], (Values[24]*Entries[24])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], (Values[25]*Entries[25])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], (Values[26]*Entries[26])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], (Values[27]*Entries[27])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)



    AG = sparse(rowAG,colAG,valAG);

    return AG;
end





function buildSparseAG_A3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner

    # Quick fix: TODO change this!


    Entries = entriesSparseG3D(k,X,Y,Z,fastconv, n ,m, l);
    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end


    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], Values[1][:]);

    # building the indices, columns and rows for the interior
    (rowAG, colAG, valAG) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], (Values[1]*Entries[1])[ :]);

     # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], Values[2][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], (Values[2]*Entries[2])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

     # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], Values[3][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], (Values[3]*Entries[3])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], (Values[4]*Entries[4])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], (Values[5]*Entries[5])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], (Values[6]*Entries[6])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], (Values[7]*Entries[7])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)





# we need to incorporate the vertices

    # for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], (Values[8]*Entries[8])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], (Values[9]*Entries[9])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], Values[10][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], (Values[10]*Entries[10])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], Values[11][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], (Values[11]*Entries[11])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], Values[12][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], (Values[12]*Entries[12])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], Values[13][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);
    # for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], (Values[13]*Entries[13])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], Values[14][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);
    # for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], (Values[14]*Entries[14])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], Values[15][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], (Values[15]*Entries[15])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], Values[16][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], (Values[16]*Entries[16])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], (Values[17]*Entries[17])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], (Values[18]*Entries[18])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], (Values[19]*Entries[19])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], (Values[20]*Entries[20])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], (Values[21]*Entries[21])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], (Values[22]*Entries[22])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], (Values[23]*Entries[23])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], (Values[24]*Entries[24])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], (Values[25]*Entries[25])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], (Values[26]*Entries[26])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], (Values[27]*Entries[27])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)



    AG = sparse(rowAG,colAG,valAG);

    return AG;
end
