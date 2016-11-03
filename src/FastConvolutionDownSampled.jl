# This file contains the types and definitions for the 
# downsampled lippmann-Schwinger equation 
using Interpolations

type FastDownSampled
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    Fast:: FastMslow
    n  :: Int64
    m  :: Int64
    downsample :: Int64
    # Sampling Matrix
    S :: SparseMatrixCSC{Float64,Int64}

    function FastDownSampled(Fast, downsample)
        # Building the Sampling matrix
        index1 = 1:downsample:Fast.n
        index2 = 1:downsample:Fast.m
        Sindex = spzeros(n,m)
        for ii = 0:round(Integer,(n-1)/downsample)
            for jj = 0:round(Integer,(m-1)/downsample)
                Sindex[1+ii*downsample,1+jj*downsample ] = 1
            end
        end

        index = find(Sindex[:])
        S = speye(n*m, n*m)
        Sampling = S[index,:];
        return new( Fast, Fast.n, Fast.m, downsample,Sampling)
    end
end


import Base.*

function *(M::FastDownSampled, b::Array{Complex128,1})
    knots = (collect(1:M.downsample:M.n), collect(1:M.downsample:M.m))
    nDown = round(Integer, (M.n-1)/M.downsample +1)
    B = reshape(b, nDown,nDown);

    itp_real = interpolate(real(B), BSpline(Quadratic(Reflect())), OnGrid())
    itp_imag = interpolate(imag(B), BSpline(Quadratic(Reflect())), OnGrid())

    interpU =     itp_real[collect(M.downsample:M.n+M.downsample-1)/M.downsample,collect(M.downsample:M.n+M.downsample-1)/M.downsample ] +
              1im*itp_imag[collect(M.downsample:M.n+M.downsample-1)/M.downsample,collect(M.downsample:M.n+M.downsample-1)/M.downsample];

    u = M.Fast*interpU[:]

    # Subsampling the wavefield
    return (M.S*u)[:]
end
