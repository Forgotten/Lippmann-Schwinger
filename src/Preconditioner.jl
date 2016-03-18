# File defining the preconditioner types

type SparsifyingPreconditioner
    Msp::SparseMatrixCSC{Complex{Float64},Int64}
    As::SparseMatrixCSC{Complex{Float64},Int64} # tranposition matrix
    MspInv
    function SparsifyingPreconditioner(Msp, As)
        tic();
        MspInv = lufact(Msp)
        println("time for the factorization was ", toc() )
        new(Msp,As, MspInv) # 
    end
end

# Encapsulation of the preconditioner in order to use preconditioned GMRES
import Base.\

function \(M::SparsifyingPreconditioner, b::Array{Complex128,1})
    # we apply the Sparsifying preconditioner 
    return M.MspInv\(M.As*b)
end
