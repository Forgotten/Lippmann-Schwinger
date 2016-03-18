# File with the functions necessary to implement the
# sparsifiying preconditioner
# Ying 2014 Sparsifying preconditioners for the Lippmann-Schwinger Equation


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
end

import Base.*

function *(M::FastM, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT

    #obtaining the middle index
    indMiddle = round(Integer, M.n-1 + (M.n+1)/2)
	  # Allocate the space for the extended B
    BExt = zeros(Complex128,M.ne, M.ne);
    # Apply spadiagm(nu) and ented by zeros
   	BExt[1:M.n,1:M.n]= reshape(M.nu.*b,M.n,M.n) ;

   	# Fourier Transform
   	BFft = fft(BExt)
   	# Component-wise multiplication
   	BFft = M.GFFT.*BFft
   	# Inverse Fourier Transform
   	BExt = ifft(BFft)

    # multiplication by omega^2
   	B = M.omega^2*(BExt[indMiddle: indMiddle+M.n-1, indMiddle:indMiddle+M.n-1]);

    return (b + B[:])
end

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
end


function *(M::FastMslow, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT

    #obtaining the middle index
    indMiddle = round(Integer, M.n-1 + (M.n+1)/2)
    # Allocate the space for the extended B
    BExt = zeros(Complex128,M.ne, M.ne);
    # Apply spadiagm(nu) and ented by zeros
    BExt[1:M.n,1:M.n]= reshape((exp(1im*M.omega*M.x).*M.nu).*b,M.n,M.n) ;

    # Fourier Transform
    BFft = fft(BExt)
    # Component-wise multiplication
    BFft = M.GFFT.*BFft
    # Inverse Fourier Transform
    BExt = ifft(BFft)

    # multiplication by omega^2
    B = M.omega^2*(BExt[indMiddle: indMiddle+M.n-1, indMiddle:indMiddle+M.n-1]);

    return (b + (exp(-1im*M.omega*M.x).*(B[:])))
end

#this is the sequantila version for sampling G
function sampleG(k,X,Y,indS, D0)
    # function to sample the Green's function at frequency k
    Gc = zeros(Complex128, length(indS), length(X))
    for i = 1:length(indS)
        ii = indS[i]
        r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
        r[ii] = 1;
        Gc[i,:] = 1im/4*hankelh1(0, k*r)*h^2;
        Gc[i,ii]= 1im/4*D0*h^2;
    end
    return Gc
end


# @everywhere function sampleG(k,X,Y,indS, D0)
#     # function to sample the Green's function at frequency k

#     R  = SharedArray(Float64, length(indS), length(X))
#     @sync @parallel for i = 1:length(indS)
#       ii = indS[i]
#       R[i,:]  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
#       R[i,ii] = 1;
#     end

#     Gc = sampleGkernelpar(k,R,h);
#     #Gc = (h^2*1im/4)*hankelh1(0, k*R)*h^2;
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
  #println(n," ")
  G = SharedArray(Complex128,n)
  rshared = convert(SharedArray, r)
  @sync begin
        @sync @parallel for ii = 1:n
            G[ii] = 1im/4*hankelh1(0, k*rshared[ii])*h^2;
        end
    end
  return sdata(G)
end

## two different versions of the same function with slight different input

@everywhere function sampleGkernelpar(k,R::Array{Float64,2},h)
  (m,n)  = size(R)
  # println("hello", n," ",m)
  G = SharedArray(Complex128,m,n)
  Rshared = convert(SharedArray, R)
  @sync begin
        for p in procs(G)
            @async remotecall_wait(sampleGkernel_shared_chunk!,p, G, Rshared,k,h)
        end
    end
  return sdata(G)
end

@everywhere function sampleGkernelpar(k,R::SharedArray{Float64,2},h)
  (m,n)  = size(R)
  # println("hello", n," ",m)
  G = SharedArray(Complex128,m,n)
  @sync begin
        for p in procs(G)
            @async remotecall_wait(sampleGkernel_shared_chunk!,p, G, R,k,h)
        end
    end
  return sdata(G)
end


# little convenience wrapper
@everywhere sampleGkernel_shared_chunk!(q,u,k,h) = sampleGkernel_chunk!(q,u,k,h, myrange(q)...)

@everywhere function sampleGkernel_chunk!(G, R,k,h, irange, jrange)
    #@show (irange, jrange)  # display so we can see what's happening
    for j in jrange, i in irange
        G[i,j] = 1im/4*hankelh1(0, k*R[i,j])*h^2;
    end
end

function entriesSparseA(k,X,Y,D0, n ,m)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex128}[]
  Indices  = Array{Int64}[]

  N = n*m;

  # computing the entries for the interior
  indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);
  indVolC = setdiff(collect(1:N),indVol);
  GSampled = sampleG(k,X,Y,indVol, D0)[:,indVolC ];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);

  # for  x = xmin, y = 0
  indFz1 = round(Integer, n*(m-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG(k,X,Y,indFz1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, [0,1,n,n+1,-n, -n+1]); #'

  # for  x = xmax, y = 0
  indFz2 = round(Integer, n*(n-1)/2 + [-1,0,n,n-1,-n, -n-1]);
  indC = setdiff(collect(1:N),indFz2);
  GSampled = sampleG(k,X,Y,indFz2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,n,n-1,-n, -n-1]); #'

  # for  y = ymin, x = 0
  indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG(k,X,Y,indFx1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,1,n,n+1, n-1]); #'

  # for  y = ymin, x = 0
  indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
  indC = setdiff(collect(1:N),indFx2);
  GSampled = sampleG(k,X,Y,indFx2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,1,-n,-n+1, -n-1]); #'

  # For the corners
  indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
  indcorner2 = round(Integer, n + [0,-1, n,n-1]);
  indcorner3 = round(Integer, n*m-n+1 + [0,1, -n,-n+1]);
  indcorner4 = round(Integer, n*m + [0,-1, -n,-n-1]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,1, n,n+1]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,-1, n,n-1]); #'

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

function referenceValsTrapRule()
    # modification to the Trapezoidal rule for logarithmic singularity
    # as explained in
    # R. Duan and V. Rokhlin, High-order quadratures for the solution of scattering problems in
    # two dimensions, J. Comput. Phys.,
    x = 2.0.^(-(0:5))[:]
    w = [1-0.892*im, 1-1.35*im, 1-1.79*im, 1- 2.23*im, 1-2.67*im, 1-3.11*im]
    return (x,w)
end


function buildSparseA(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       D0::Complex128, n::Int64 ,m::Int64)
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);


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

function entriesSparseG(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       D0::Complex128, n::Int64 ,m::Int64)
  # function to compute the entried of G, inside the volume, at the boundaries
  # and at the corners. This allows us to compute A*G in O(n) time instead of
  # O(n^2)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex128}[]

  N = n*m;

  # computing the entries for the interior
  indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);
  GSampled = sampleG(k,X,Y,indVol, D0)[:,indVol];

  push!(Entries,GSampled);

  # for  x = xmin, y = 0
  indFz1 = round(Integer, n*(m-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
  GSampled = sampleG(k,X,Y,indFz1, D0)[:,indFz1];

  push!(Entries,GSampled);

  # for  x = xmax, y = 0
  indFz2 = round(Integer, n*(n-1)/2 + [-1,0,n,n-1,-n, -n-1]);
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
                       D0::Complex128, n::Int64 ,m::Int64)
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);
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

function buildGConv(x,y,h,n,m,D0,k)

    # build extended domain
    xe = collect((x[1]-(n-1)*h):h:(x[end]+(n-1)*h));
    ye = collect((y[1]-(m-1)*h):h:(y[end]+(m-1)*h));

    Xe = repmat(xe, 1, 3*m-2);
    Ye = repmat(ye', 3*n-2,1);

    R = sqrt(Xe.^2 + Ye.^2);
    # to avoid evaluating at the singularity
    indMiddle = round(Integer, m-1 + (m+1)/2)
    # we modify R to remove the zero (so we don't )
    R[indMiddle,indMiddle] = 1;
    # sampling the Green's function
    Ge = 1im/4*hankelh1(0, k*R)*h^2;
    # modiyfin the diagonal with the quadrature
    # modification
    Ge[indMiddle,indMiddle] = 1im/4*D0*h^2;

return Ge

end

function buildGConvPar(x,y,h,n,m,D0,k)

    # build extended domain
    xe = collect((x[1]-(n-1)*h):h:(x[end]+(n-1)*h));
    ye = collect((y[1]-(m-1)*h):h:(y[end]+(m-1)*h));

    Xe = repmat(xe, 1, 3*m-2);
    Ye = repmat(ye', 3*n-2,1);

    R = sqrt(Xe.^2 + Ye.^2);
    # to avoid evaluating at the singularity
    indMiddle = round(Integer, m-1 + (m+1)/2)
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


