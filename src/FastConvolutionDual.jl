# function to build the fast application of the dual Lippmann-Schwinger equation
# we suppose u(x) = \int_D G(x,y) \sigma(y) dy
# and we solve ( \Delta + omega^2(1+q)) u = - omega^2q u^{inc}



type FastMDual
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
    function FastMDual(GFFT,nu,ne,me,n,m,k; quadRule::String = "trapezoidal")
      return new(GFFT,nu,ne,me,n, m, k, quadRule)
    end
end

import Base.*
import Base.A_mul_B!
import Base.size

function size(M::FastMDual, dim::Int64)
  # function to returns the size of the underliying matrix (M.m*M.n)^2
  return M.m*M.n
end

function *(M::FastMDual, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT
    # dummy function to call fastconvolution
    return fastconvolution(M,b)
end

function A_mul_B!(Y,
                  M::FastMDual,
                  V)
    # in place matrix matrix multiplication
    @assert(size(Y) == size(V))
    # print(size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = M*V[:,ii]
    end
end



@inline function fastconvolution(M::FastMDual, b::Array{Complex128,1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT

    # computing omega^2 nu G*(b)
    B = M.omega^2*(M.nu.*FFTconvolution(M,b))

    # returning -b + omega^2(nu G*b)
    return (-b + B[:])
end


@inline function FFTconvolution(M::FastMDual, b::Array{Complex128,1})
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


function buildFastConvolutionDual(x::Array{Float64,1},y::Array{Float64,1},
                              h::Float64,k,nu::Function; quadRule::String = "trapezoidal")

  if quadRule == "trapezoidal"

    (ppw,D) = referenceValsTrapRule();
    D0      = D[round(Int,k*h)];
    (n,m) = length(x), length(y)
    Ge    = buildGConv(x,y,h,n,m,D0,k);
    GFFT  = fft(Ge);
    X = repmat(x, 1, m)[:]
    Y = repmat(y', n,1)[:]

    return FastMDual(GFFT,nu(X,Y),2*n-1,2*m-1,n, m, k);

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

        S = sqrt.(KX.^2 + KY.^2);

        GFFT = Gtruncated2D(L, k, S)
        return FastMDual(GFFT, nu(X,Y), 4*n, 4*m,
                     n, m, k , quadRule="Greengard_Vico");
      else
        # kx = (-2*(n-1):1:2*(n-1) )/4;
        # ky = (-2*(m-1):1:2*(m-1) )/4;

        # KX = (2*pi/Lp)*repmat(kx, 1, 4*m-3);
        # KY = (2*pi/Lp)*repmat(ky', 4*n-3,1);

        # S = sqrt.(KX.^2 + KY.^2);

        # GFFT = Gtruncated2D(L, k, S)

        # return FastM(GFFT,nu(X,Y),4*n-3,4*m-3,
        #              n,m, k,quadRule = "Greengard_Vico");

        kx = (-2*n:1:2*n-1);
        ky = (-2*m:1:2*m-1);

        KX = (2*pi/Lp)*repmat( kx, 1,4*m);
        KY = (2*pi/Lp)*repmat(ky',4*n,  1);

        S = sqrt.(KX.^2 + KY.^2);

        GFFT = Gtruncated2D(L, k, S)

        return FastMDual(GFFT,nu(X,Y),4*n,4*m,
                     n,m, k,quadRule = "Greengard_Vico");


    end
  end
end

##########################################################################
########  Function fo build the preconditioner for the  ##################
########        Dual Lippmann-Schwinger equation    ######################
##########################################################################


function buildGSparseAandG(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          D0::Complex128, n::Int64 ,m::Int64, nu::Function ; method::String = "normal")

    nuArray = nu(X,Y)
    Ind = reshape(collect(1:n*m),n,m);

    if method=="normal"
      (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);
    elseif method == "randomized"
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    Entries = entriesSparseG(k,X,Y,D0, n ,m);

    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1][:],
                                       Indices[1][:], Values[1][:]);

    (rowG, colG, valG) = createIndices(Ind[2:end-1,2:end-1][:],
                                       Indices[1][:],Values[1][:], Entries[1],nuArray);


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


    (Row, Col, Val) = createIndices(Ind[1,2:end-1][:],
                                    Indices[2][:],Values[2][:], Entries[2],nuArray);

    rowG = vcat(rowG,Row);
    colG = vcat(colG,Col);
    valG = vcat(valG,Val);


    (Row, Col, Val) = createIndices(Ind[end,2:end-1][:],
                                    Indices[3][:], Values[3][:], Entries[3],nuArray);
    rowG = vcat(rowG,Row)
    colG = vcat(colG,Col)
    valG = vcat(valG,Val)

    (Row, Col, Val) = createIndices(Ind[2:end-1,1][:],
                                    Indices[4][:], Values[4][:], Entries[4],nuArray);

    rowG = vcat(rowG,Row)
    colG = vcat(colG,Col)
    valG = vcat(valG,Val)

    (Row, Col, Val) = createIndices(Ind[2:end-1,end][:],
                                    Indices[5][:], Values[5][:], Entries[5],nuArray);
    rowG = vcat(rowG,Row)
    colG = vcat(colG,Col)
    valG = vcat(valG,Val)

    (Row, Col, Val) = createIndices(Ind[1,1],
                                    Indices[6][:], Values[6][:], Entries[6],nuArray);
    rowG = vcat(rowG,Row)
    colG = vcat(colG,Col)
    valG = vcat(valG,Val)

    (Row, Col, Val) = createIndices(Ind[end,1],
                                    Indices[7][:], Values[8][:], Entries[8],nuArray);

    rowG = vcat(rowG,Row)
    colG = vcat(colG,Col)
    valG = vcat(valG,Val)

    (Row, Col, Val) = createIndices(Ind[1,end],
                                    Indices[8][:], Values[8][:], Entries[7],nuArray);

    rowG = vcat(rowG,Row)
    colG = vcat(colG,Col)
    valG = vcat(valG,Val)

    (Row, Col, Val) = createIndices(Ind[end,end],
                                    Indices[9][:], Values[9][:], Entries[9],nuArray);

    rowG = vcat(rowG,Row)
    colG = vcat(colG,Col)
    valG = vcat(valG,Val)

    ADGs = sparse(rowG,colG,valG);

    return (A, ADGs)

end


function createIndices(row::Array{Int64,1}, col::Array{Int64,1},
                       valA::Array{Complex128,1},
                       valG::Array{Complex128,2}, nu::Array{Float64,1})
  # function to create the indices for a sparse matrix
  @assert length(col) == length(valA)
  nn = length(col);
  mm = length(row);

  ll = length(valA)

  Row = kron(row, ones(Int64, nn));
  Col = kron(ones(Int64,mm), col) + Row;
  Val = nu[Col].*(kron(ones(Int64,mm), valA))

  # multiplying by the blocks of G.
  for ii = 1:mm
    Val[(ii-1)*ll+1:ii*ll] = (Val[(ii-1)*ll+1:ii*ll].')*valG
  end


  return (Row,Col,Val)
end

function createIndices(row::Int64, col::Array{Int64,1},
                       valA::Array{Complex128,1},
                       valG::Array{Complex128,2}, nu::Array{Float64,1})

  @assert length(col) == length(valA)
  nn = length(col);
  mm = 1;
  ll = length(valA)

  Row = kron(row, ones(Int64, nn));
  Col = kron(ones(Int64,mm), col) + Row;
  Val = nu[Col].*(kron(ones(Int64,mm), valA))

  print(size(Val))

  # multiplying by the blocks of G.
  for ii = 1:mm
    Val[(ii-1)*ll+1:ii*ll] = (Val[(ii-1)*ll+1:ii*ll].')*valG
  end

  return (Row,Col,Val)
end


# function sparsifiedMatrix()

#     As = buildSparseA(k,X,Y,D0, n ,m);


