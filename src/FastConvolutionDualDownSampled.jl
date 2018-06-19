# This file contains the types and definitions for the 
# downsampled lippmann-Schwinger equation 
using Interpolations

type FastDualDownSampled
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    # Fast:: FastMslow
    Fast:: FastMslowDual
    n  :: Int64
    m  :: Int64
    # downsampling factor in each dimension
    downsampleX :: Int64
    downsampleY :: Int64
    # Sampling Matrix
    S :: SparseMatrixCSC{Float64,Int64}

    function FastDualDownSampled(Fast, downsampleX, downsampleY)
        # Building the Sampling matrix
        index1 = 1:downsampleX:Fast.n
        index2 = 1:downsampleY:Fast.m
        Sindex = spzeros(n,m)
        for ii = 0:round(Integer,(n-1)/downsampleX)
            for jj = 0:round(Integer,(m-1)/downsampleY)
                Sindex[1+ii*downsampleX,1+jj*downsampleY ] = 1
            end
        end

        index = find(Sindex[:])
        S = speye(n*m, n*m)
        Sampling = S[index,:];
        return new( Fast, Fast.n, Fast.m, downsampleX, downsampleY,Sampling)
    end
end

# we need to overload these functions for gmres
import Base.*
import Base.A_mul_B!
import Base.size

function size(M::FastDualDownSampled, dim::Int64)
  # function to returns the size of the underliying matrix i.e. the rank of M.S
  return size(M.S,1)
end


function A_mul_B!(Y,
                  M::FastDualDownSampled,
                  V)
    # in place matrix matrix multiplication
    @assert(size(Y) == size(V))
    # print(size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = M*V[:,ii]
    end
end


function *(M::FastDualDownSampled, b::Array{Complex128,1})
    knots = (collect(1:M.downsampleX:M.n), collect(1:M.downsampleY:M.m))
    nDown = round(Integer, (M.n-1)/M.downsampleX +1)
    mDown = round(Integer, (M.m-1)/M.downsampleY +1)
    B = reshape(b, nDown,mDown);

    itp_real = interpolate(real(B), BSpline(Quadratic(Reflect())), OnGrid())
    itp_imag = interpolate(imag(B), BSpline(Quadratic(Reflect())), OnGrid())

    interpU =     itp_real[collect(M.downsampleX:M.n+M.downsampleX-1)/M.downsampleX,
                           collect(M.downsampleY:M.m+M.downsampleY-1)/M.downsampleY] +
              1im*itp_imag[collect(M.downsampleX:M.n+M.downsampleX-1)/M.downsampleX,
                           collect(M.downsampleY:M.m+M.downsampleY-1)/M.downsampleY];

    u = M.Fast*interpU[:]
    size(u)
    # Subsampling the wavefield
    return (M.S*u)[:]
end
