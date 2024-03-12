module games
    using LinearAlgebra
    using TensorOperations
    
    !isdefined(Main, :nsboxes) ? error("First import the nsboxes module") : nothing
    using ..nsboxes
    !isdefined(Main, :wirings) ? error("First import the wiring module") : nothing
    using ..wirings
    
    #export 

    # Utility functions

    H_Bin(p) = -p*log2(p) - (1-p)*log2(1-p) # Binary entropy function


    # Games
    # ----------------------------------------------------------------
    # CHSH game

    function CHSH_score_generator(s1 = 1, s2 = 1, s3 = 1, s4 = -1; batched=false)
        @assert s1 ∈ [-1,1] && s2 ∈ [-1,1] && s3 ∈ [-1,1] && s4 ∈ [-1,1] "Sign for each term in CHSH Bell inequality must be either -1 or 1"
        @assert s1*s2*s3*s4 == -1 "Parity condition for CHSH inequality not satisfied; Only odd number of terms can have a minus sign"

        function abs_CHSH_score(nsjoint::Array{Float64,4}) # nsjoint is a 2x2x2x2 array
            """CHSH functional S"""
            E_xy = ((2.0 .* reduce(+, nsjoint[o,o,:,:] for o in 1:2)) .- 1.0) #  ProbBellCorrelator
            return abs(s1*E_xy[1,1] + s2*E_xy[1,2] + s3*E_xy[2,1] + s4*E_xy[2,2])
        end

        function batched_abs_CHSH_score(R::Array{Float64,5}) # R is a 2x2x2x2xn tensor
            """CHSH functional S"""
            E_xyn = ((2.0 .* reduce(+, R[o,o,:,:,:] for o in 1:2)) .- 1.0) #  ProbBellCorrelator
            #E_xyn = (@tensor T1[k,l,b] := (2.0*R[i,i,k,l,b])) .- 1.0
            return abs.(s1*E_xyn[1,1,:] .+ s2*E_xyn[1,2,:] .+ s3*E_xyn[2,1,:] .+ s4*E_xyn[2,2,:])
        end
        
        return batched ? batched_abs_CHSH_score : abs_CHSH_score
    end

    function canonical_CHSH_score(P::Array{Float64,4}) 
        #unbatched version
        E_xyn = (@tensor T1[k,l] := (2.0*P[i,i,k,l])) .- 1.0
        return abs(E_xyn[1,1] + E_xyn[1,2] + E_xyn[2,1] - E_xyn[2,2])
    end
    function canonical_CHSH_score(P::Array{Float64,5}) 
        #batched version
        E_xyn = (@tensor T1[k,l,b] := (2.0*P[i,i,k,l,b])) .- 1.0
        return abs.(E_xyn[1,1,:] .+ E_xyn[1,2,:] .+ E_xyn[2,1,:] .- E_xyn[2,2,:])
    end
    canonical_CHSH_score(W::Matrix{<:Real}, P_mat::Matrix{<:Float64}, Q_mat::Matrix{<:Float64}) = canonical_CHSH_score( wirings.tensorized_boxproduct(W, P_mat, Q_mat) )
    canonical_CHSH_score(W::Matrix{<:Real}, P_joint::Array{<:Float64,4}, Q_joint::Array{<:Float64,4}) = canonical_CHSH_score(W, wirings.convert_nsjoint_to_matrixbox(P_joint), wirings.convert_nsjoint_to_matrixbox(Q_joint))
    

    # ----------------------------------------------------------------
    # IC Mutual Information game
    P_RAC_lossless_vanDam_CHSH_coeffs = zeros(2,2)
    for i in 0:1
        for j in 0:1
            if j == 0
                P_RAC_lossless_vanDam_CHSH_coeffs[j+1,i+1] = (1/(2^2))*2
            elseif i == 0 && j==1
                P_RAC_lossless_vanDam_CHSH_coeffs[j+1,i+1] = (1/(2^2))*(2^(j))
            elseif i != 0 && j>=1 && j<=2-i
                P_RAC_lossless_vanDam_CHSH_coeffs[j+1,i+1] = (1/(2^2))*( (-1)^(j == 2-i) )*(2^(j))
            end
        end
    end

    
    function MutInfo_IC_vanDam_score(FullBoxJoint::Array{Float64, 4}; e_c=0.01)
        # non-batched version; FullBoxJoint is a 2x2x2x2 tensor
        # e_c = Binary symmetric noise channel bias parameter
        
        #BoxBiases = (2 .* sum(FullBoxJoint[o,o,:,:] for o in 1:size(FullBoxJoint)[1])) .- 1 # 2*P(A=B|X,Y) - 1
        BoxBiases = (@tensor T1[k,l] := (2.0*FullBoxJoint[i,i,k,l])) .- 1.0 # 2*P(A=B|X,Y) - 1 for each of the n columns. This is a 2x2 tensor of inputs x,y

        #Note that sum() does not drop the dimension!; I checked that neither tensoroperations nor for-loops are faster than the folllowing
        P_RAC_vanDam = 1/2 .+ ((e_c/2)*sum(P_RAC_lossless_vanDam_CHSH_coeffs .* BoxBiases, dims=1)) # Only sum of j (=x=first index) -> outputs a 1x2xn tensor
        return 2 - sum(H_Bin.(P_RAC_vanDam), dims=2)[] # Sum over j (= y = second index) -> outputs a scalar ([] converts 1x1 array to scalar)
    end

    function MutInfo_IC_vanDam_score(batched_FullBoxJoint::Array{Float64, 5}; e_c=0.01)
        # batched version; FullBoxJoint is a 2x2x2x2xn tensor
        # e_c = Binary symmetric noise channel bias parameter
        
        #BoxBiases = (2 .* sum(FullBoxJoint[o,o,:,:,:] for o in 1:size(FullBoxJoint)[1])) .- 1 # 2*P(A=B|X,Y) - 1 for each of the n columns. This is a 2x2xn tensor of inputs x,y
        BoxBiases = (@tensor T1[k,l,b] := (2.0*batched_FullBoxJoint[i,i,k,l,b])) .- 1.0 # 2*P(A=B|X,Y) - 1 for each of the n columns. This is a 2x2xn tensor of inputs x,y

        #Note that sum() does not drop the dimension!
        P_RAC_vanDam = 1/2 .+ ((e_c/2)*sum(reshape(P_RAC_lossless_vanDam_CHSH_coeffs, size(P_RAC_lossless_vanDam_CHSH_coeffs)...,1) .* BoxBiases, dims=1)) # Only sum of j (=x=first index) -> outputs a 1x2xn tensor
        return 2 .- reshape(sum(H_Bin.(P_RAC_vanDam), dims=2), :) # Sum over j (= y = second index) -> outputs a n tensor
    end

    function MutInfo_IC_vanDam_score(W::Matrix{<:Real}, P_mat::Matrix{<:Float64}, Q_mat::Matrix{<:Float64}) # W is a 32xn tensor, P and Q are 4x4 matrices
        return MutInfo_IC_vanDam_score( wirings.tensorized_boxproduct(W, P_mat, Q_mat); e_c=0.01 ) # tensorized_boxproduct() gives 2x2x2x2xn tensor
    end

    MutInfo_IC_vanDam_score(W::Matrix{<:Real}, P_joint::Array{<:Float64,4}, Q_joint::Array{<:Float64,4}) = MutInfo_IC_vanDam_score(W, wirings.convert_nsjoint_to_matrixbox(P_joint), wirings.convert_nsjoint_to_matrixbox(Q_joint))
    MutInfo_IC_vanDam_score(W::Vector{<:Real}, P_mat::Matrix{Float64}, Q_mat::Matrix{Float64}) = MutInfo_IC_vanDam_score( wirings.reduc_tensorized_boxproduct(reshape(W,32,1), P_mat, Q_mat); e_c=0.01 )

    # ----------------------------------------------------------------
    # IC Mutual Information game - Pawloski et al. 2009

    function Original_IC_Bound_score(FullBoxJoint::Array{Float64, 4})
        # non-batched version; FullBoxJoint is a 2x2x2x2 tensor
        # e_c = Binary symmetric noise channel bias parameter
        
        #BoxBiases = (2 .* sum(FullBoxJoint[o,o,:,:] for o in 1:size(FullBoxJoint)[1])) .- 1 # 2*P(A=B|X,Y) - 1
        BoxBiases = (@tensor T1[k,l] := (2.0*FullBoxJoint[i,i,k,l])) .- 1.0 # 2*P(A=B|X,Y) - 1 for each of the n columns. This is a 2x2 tensor of inputs x,y
        return abs(BoxBiases[1,1] + BoxBiases[2,1])^2 + abs(BoxBiases[1,2] - BoxBiases[2,2])^2
    end

    function Original_IC_Bound_score(batched_FullBoxJoint::Array{Float64, 5})
        # batched version; FullBoxJoint is a 2x2x2x2xn tensor
        # e_c = Binary symmetric noise channel bias parameter
        
        #BoxBiases = (2 .* sum(FullBoxJoint[o,o,:,:,:] for o in 1:size(FullBoxJoint)[1])) .- 1 # 2*P(A=B|X,Y) - 1 for each of the n columns. This is a 2x2xn tensor of inputs x,y
        BoxBiases = (@tensor T1[k,l,b] := (2.0*batched_FullBoxJoint[i,i,k,l,b])) .- 1.0 # 2*P(A=B|X,Y) - 1 for each of the n columns. This is a 2x2xn tensor of inputs x,y
        return abs.(BoxBiases[1,1,:] + BoxBiases[2,1,:]).^2 + abs.(BoxBiases[1,2,:] - BoxBiases[2,2,:]).^2
    end

    Original_IC_Bound_score(W::Matrix{<:Real}, P_mat::Matrix{Float64}, Q_mat::Matrix{Float64}) = Original_IC_Bound_score( wirings.tensorized_boxproduct(W, P_mat, Q_mat) )
    Original_IC_Bound_score(W::Matrix{<:Real}, P_joint::Array{<:Float64,4}, Q_joint::Array{<:Float64,4}) = Original_IC_Bound_score(W, wirings.convert_nsjoint_to_matrixbox(P_joint), wirings.convert_nsjoint_to_matrixbox(Q_joint))
    
    Original_IC_Bound_score(W::Vector{<:Real}, P_mat::Matrix{Float64}, Q_mat::Matrix{Float64}) = Original_IC_Bound_score( wirings.reduc_tensorized_boxproduct(reshape(W,32,1), P_mat, Q_mat) )
    

end