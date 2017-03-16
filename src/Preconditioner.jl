# File defining the preconditioner types

type SparsifyingPreconditioner
    Msp::SparseMatrixCSC{Complex{Float64},Int64}
    As::SparseMatrixCSC{Complex{Float64},Int64} # tranposition matrix
    MspInv
    solverType::String
    function SparsifyingPreconditioner(Msp, As; solverType::String="UMPFACK")
        tic();

        if solverType == "UMPFACK"
            MspInv = lufact(Msp)

        elseif solverType == "MKLPardiso"
            # using MKLPardiso from Julia Sparse (only shared memory)
            println("factorizing the matrix using MKL Pardiso")
            MspInv = MKLPardisoSolver();
            set_nprocs!(MspInv, 4)
            #setting the type of the matrix
            set_matrixtype!(MspInv,3)
            # setting we are using a transpose
            set_iparm!(MspInv,12,2)
            # setting the factoriation phase
            set_phase!(MspInv, 12)
            X = zeros(Complex128, size(Msp)[1],1)
            # factorizing the matrix
            pardiso(MspInv,X, Msp,X)
            # setting phase and parameters to solve and transposing the matrix
            # this needs to be done given the different C and Fortran convention
            # used by Pardiso (C convention) and Julia (Fortran Convention)
            set_phase!(MspInv, 33)
            set_iparm!(MspInv,12,2)
        end

        println("time for the factorization was ", toc() )
        new(Msp,As, MspInv,solverType) #
    end
end

# Encapsulation of the preconditioner in order to use preconditioned GMRES
import Base.\

function \(M::SparsifyingPreconditioner, b::Array{Complex128,1})
    # we apply the Sparsifying preconditioner
    rhs = (M.As*b)
    if M.solverType == "UMPFACK"
        return M.MspInv\rhs
    elseif M.solverType == "MKLPardiso"
        set_phase!(M.MspInv, 33)
        u = zeros(Complex128,length(rhs),1)
        pardiso(M.MspInv, u, M.Msp, rhs)
    end
end
