!isdefined(Main, :nsboxes) ? (include("nsboxes.jl"); using .nsboxes) : nothing
!isdefined(Main, :conditions) ? (include("conditions.jl"); using .conditions) : nothing

using Distances

global_eps_tol = 1e-10 #Global tolerance for approximate comparisons

function NS_Conditions(;scenario=(2,2,2,2))
    function all_NS_constraints(full_joint::Array{Float64, 4})
        full_A_marginals = sum(full_joint, dims=2)
        full_B_marginals = sum(full_joint, dims=1)

        global_norm = all(abs.(sum(full_A_marginals, dims=(1,2)) .- 1.0) .< global_eps_tol) #Normalization constraint on P(a,b|x,y) for all (x,y) pairs
        NS_B_to_A = all(abs.(full_A_marginals .-full_A_marginals[:,:,:,1:1]) .< global_eps_tol) #No-signaling constraints for marginals of A
        NS_A_to_B = all(abs.(full_B_marginals .-full_B_marginals[:,:,1:1,:]) .< global_eps_tol) #No-signaling constraints for marginals of B
        return global_norm && NS_B_to_A && NS_A_to_B
    end

    return conditions.GeneralizedCorrelationCondition(;scenario=scenario, functional=all_NS_constraints, compare_operator=(==), bound=true, setBounds=missing)
end


#---------------- Correlators: ----------------
#Einsum does not support Num: ProbBellCorrelator(full_joint::Array{Float64, 4}) = (2.0 .* (@einsum P_guess_xy[x,y] := full_joint[o,o,x,y]) .- 1.0))
ProbBellCorrelator(full_joint::Array{Float64, 4}) = (2.0 .* reduce(+, full_joint[o,o,:,:] for o in 1:min(size(full_joint,1),size(full_joint,2))) .- 1.0)
ProbBellCorrelator(nsbox::nsboxes.NSBox) = ProbBellCorrelator(nsboxes.reconstructFullJoint(nsbox)) #TODO: Compute more efficient expression for NS-CG representation via symbolics.jl
MarginalProbBellCorrelators(full_joint::Array{Float64, 4}) = (x=dropdims(sum(full_joint[1:1, :, :, 1:1] - full_joint[2:2, :, :, 1:1], dims=2), dims=(1,2)), y=dropdims(sum(full_joint[:, 1:1, 1:1, :] - full_joint[:, 2:2, 1:1, :], dims=1), dims=(1,2)) )
MarginalProbBellCorrelators(nsbox::nsboxes.NSBox) = MarginalProbBellCorrelators(nsboxes.reconstructFullJoint(nsbox)) #TODO: Compute more efficient expression for NS-CG representation via symbolics.jl

function CHSH_Inequality(;s1 = 1, s2 = 1, s3 = 1, s4 = -1)
    """Constructs a CHSH Bell inequality condition from the coefficients s1, s2, s3, s4
    """
    @assert s1 ∈ [-1,1] && s2 ∈ [-1,1] && s3 ∈ [-1,1] && s4 ∈ [-1,1] "Sign for each term in CHSH Bell inequality must be either -1 or 1"
    @assert s1*s2*s3*s4 == -1 "Parity condition for CHSH inequality not satisfied; Only odd number of terms can have a minus sign"
    
    function abs_S_CHSH(nsjoint::Array{Float64, 4}) #::nsboxes.NSBox
        """CHSH Bell inequality functional S for a NSBox
        """
        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
        return abs(s1*E_xy[1,1] + s2*E_xy[1,2] + s3*E_xy[2,1] + s4*E_xy[2,2])
    end

    return conditions.GeneralizedCorrelationCondition(;scenario=(2,2,2,2), functional=abs_S_CHSH, compare_operator=(<=), bound=missing, setBounds=Dict(:L => 2, :Q => 2*sqrt(2), :NS => 4))
end

function Uffink_Bipartite_2222_Inequality()
    """Constructs the quadratic Uffink bipartite Bell inequality in it's canonical form
    """
    function S_Uffink_Bipartite_2222(nsjoint::Array{Float64, 4}) #::nsboxes.NSBox
        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
        return abs(E_xy[1,2] + E_xy[2,1])^2 + abs(E_xy[1,1] - E_xy[2,2])^2
    end

    return conditions.GeneralizedCorrelationCondition(;scenario=(2,2,2,2), functional=S_Uffink_Bipartite_2222, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 4, :NS => 8)) #:L => 4,
end


function NPA_TLM_Criterion()
    """Constructs the TLM criterion (+ refinement by NPA)
    """
    function npa_tlm_functional(nsjoint::Array{Float64, 4}) #::nsboxes.NSBox
        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
        marg_E = MarginalProbBellCorrelators(nsjoint)
        D_xy = (E_xy .- (marg_E.x .* marg_E.y)) ./ sqrt.((1 .- marg_E.x) .* (1 .- marg_E.y))
        #@show D_xy
        try
            return abs(asin(D_xy[1,2]) + asin(D_xy[2,1]) + asin(D_xy[1,1]) - asin(D_xy[2,2]))
        catch
            @warn "NPA_TLM_Criterion: asin(D_xy) not defined for some D_xy; returning NaN"
            return NaN     
        end
    end
    return conditions.GeneralizedCorrelationCondition(;scenario=(2,2,2,2), functional=npa_tlm_functional, compare_operator=(<=), bound=pi, setBounds=Dict(:Q => pi))
end



function I3322_Inequality()

    function I3322(nsjoint_3322::Array{Float64, 4}) #::nsboxes.NSBox
        nsbox_3322 = nsboxes.NSBox((3,3,2,2), nsjoint_3322)
        I_3322_GC = Dict("marginals_A"=>[-1,0,0], "marginals_B"=>[-2,-1,0], "joints"=>[1 1 1;1 1 -1;1 -1 0])
        return dot(I_3322_GC["marginals_A"], nsbox_3322.marginals_vec_A) + dot(I_3322_GC["marginals_B"], nsbox_3322.marginals_vec_B) + sum(I_3322_GC["joints"] .* nsbox_3322.joints_mat)
    end
    
    return conditions.GeneralizedCorrelationCondition(;scenario=(3,3,2,2), functional=I3322, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 0))
end

function Original_IC_Bound()
    function S_IC(nsjoint::Array{Float64, 4}) #::nsboxes.NSBox
        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
        return abs(E_xy[1,1] + E_xy[2,1])^2 + abs(E_xy[1,2] - E_xy[2,2])^2
    end
    return conditions.GeneralizedCorrelationCondition(;scenario=(2,2,2,2), functional=S_IC, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 4))
end


function Correlated_Inputs_IC_Bound(;ϵ)
    @assert -1 <= ϵ <= 1 "Input correlation ϵ must be in [-1,1]" 
    function S_IC(nsjoint::Array{Float64, 4}) #::nsboxes.NSBox
        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
        return ((1+ϵ)*E_xy[1,1] + (1-ϵ)*E_xy[2,1])^2 + (1-ϵ^2)*(E_xy[1,2] - E_xy[2,2])^2
    end
    return conditions.GeneralizedCorrelationCondition(;scenario=(2,2,2,2), functional=S_IC, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 4))
end

#-----------
#Binary entropy function
function H_bin(p)
    if abs(p) < 10^(-5)
        p = abs(p)
    end
    return -p*log2(p) - (1-p)*log2(1-p)
end

function IC_Success_P_GeneralVanDam(nsjoint::Array{Float64, 4}; e_c::Real, return_E_xy::Bool=false)
    n = nsbox.scenario[1]
    E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
    
    Psuccess = []
    push!(Psuccess, 1/2 + (e_c/2)*((1/(2^n))*(2*E_xy[0+1,0+1] + sum(2^j * E_xy[j+1,0+1] for j in 1:n-1))))
    for i in 1:n-1 #1-indexing shift
        push!(Psuccess, 1/2 + (e_c/2)*((1/(2^n))*(2*E_xy[0+1,i+1] + sum((-1)^(j==(n-i)) * 2^j * E_xy[j+1,i+1] for j in 1:n-i))))
    end

    if return_E_xy
        return Psuccess, E_xy
    else
        return Psuccess
    end
end

function Generalized_Original_IC_Bound(;n=2)
    
    function S_IC(nsjoint::Array{Float64, 4})
        @assert (size(nsjoint,1),size(nsjoint,2)) == (n, n)
        @assert (size(nsjoint,3),size(nsjoint,4)) == (2, 2)

        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator

        sum_of_quadratic_terms = ((1/(2))*(2*E_xy[0+1,0+1] + sum(2^j * E_xy[j+1,0+1] for j in 1:n-1)))^2
        for i in 1:n-1 #1-indexing shift
            sum_of_quadratic_terms += ((1/(2))*(2*E_xy[0+1,i+1] + sum((-1)^(j==(n-i)) * 2^j * E_xy[j+1,i+1] for j in 1:n-i)))^2
        end
        return sum_of_quadratic_terms 
    end

    return conditions.GeneralizedCorrelationCondition(;scenario=(n,n,2,2), functional=S_IC, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 4^(n-1)))
end


# ------------------------------------ #
# Numerical, Redundant Information bound (see Yu_Scarani_2022)
# ------------------------------------ #


function condit_and_B_marginal_RAC_guess_probabilities(E_xy::Array{Float64, 2}; n, e_c)
    @assert n == 2

    f(in_vec) = in_vec[0+1] ⊻ in_vec[1+1] # mod(n - 1 - sum(prod( (α_vec[0+1] + α_vec[l+1] for l in 1:i )  ) for i in 1:n-1) ,2)
    h(in_vec) = in_vec[0+1]

    condit_probabs = Array{Float64}(undef, 2^n, 2, n) #P(α_vec | B_i, i = β)
    B_marginal_probabs = zeros(2, n) #P(B_i, i = β)
    for β in 0:n-1
        for B_i_val in 0:1
            for (α_idx, α_vec) in enumerate(Iterators.product(fill(0:1, n)...))
                nominator = 1/2 + ((e_c/2)*sum( (f(α_vec) == j) * (-1)^(h(α_vec) == B_i_val) * E_xy[j+1,β+1] for j in 0:n-1))
                denominator = 1/2 + ((e_c/2) * (1/(2^n)) * sum( sum( (f(k_vec) == j) * (-1)^(h(k_vec) == B_i_val) * E_xy[j+1,β+1] for j in 0:n-1) for k_vec in Iterators.product(fill(0:1, n)...)))
                condit_probabs[α_idx, B_i_val+1, β+1] = (1/(2^n)) * nominator/denominator
                B_marginal_probabs[B_i_val+1, β+1] += (1/(2^n))*nominator
            end
        end
    end

    return condit_probabs, B_marginal_probabs #2^n x 2 x n array and 2 x n matrix, representing P(α_vec | B_i = b_i, i = β) (with different β in 2nd dimension) and P(B_i | i = β)
end

H_Shannon(p_vec::Vector{Float64}) = -sum(p*log2(p) for p in p_vec)

function total_mutual_info(condit_probabs::Array{Float64, 3}, B_marginal_probabs::Matrix{Float64}; n)    
    # I(α_vec : B_i | i = β) = 1 - sum_{b_i} P(B_i = b_i | i = β)*H(α_vec | B_i = b_i, i = β)
    return sum(n - sum(B_marginal_probabs[B_i_val+1, β+1] * H_Shannon(condit_probabs[:, B_i_val+1, β+1]) for B_i_val in 0:1) for β in 0:n-1)
end


binary_convex_mixture(p::Vector{Float64}, q::Vector{Float64}; λ::Float64) = λ*p + (1-λ)*q

function total_redundant_info(condit_probabs::Array{Float64, 3}; n)
    #Starting point = condit probabs P(α_vec | B_i, i=β)

    optim_sample_resolution = 1e3 #or 20000

    mixtures_B_2 = [binary_convex_mixture(condit_probabs[:, 0+1, 2], condit_probabs[:, 1+1, 2]; λ) for λ in range(0.0, 1.0, step=1/optim_sample_resolution)]
    closest_condit_probabs_1_to_2 = [argmin(mixture -> kl_divergence(condit_probabs[:, B_1_val+1, 1], mixture), mixtures_B_2) for B_1_val in 0:1]
    projected_info_12 = sum(condit_probabs[α_idx, B_1_val+1, 1] * log(closest_condit_probabs_1_to_2[B_1_val+1][α_idx] / (1/2^n) ) for B_1_val in 0:1, α_idx in 1:2^n)

    mixtures_B_1 = [binary_convex_mixture(condit_probabs[:, 0+1, 1], condit_probabs[:, 1+1, 1]; λ) for λ in range(0.0, 1.0, step=1/optim_sample_resolution)]
    closest_condit_probabs_2_to_1 = [argmin(mixture -> kl_divergence(condit_probabs[:, B_2_val+1, 2], mixture), mixtures_B_1) for B_2_val in 0:1]
    projected_info_21 = sum(condit_probabs[α_idx, B_2_val+1, 2] * log(closest_condit_probabs_2_to_1[B_2_val+1][α_idx] / (1/2^n) ) for B_2_val in 0:1, α_idx in 1:2^n)

    return min(projected_info_12, projected_info_21)
end

function RedundantInfo_IC_Bound(;n=2, e_c=0.001)
    """ Assume (n,n,2,2) scenario and uniform distributed input data α_vec at A. 
    """

    function S_IC(nsjoint::Array{Float64, 4})
        @assert (size(nsjoint,1),size(nsjoint,2)) == (n, n)
        @assert (size(nsjoint,3),size(nsjoint,4)) == (2, 2)

        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator

        condit_probabs, B_marginals_probabs = condit_and_B_marginal_RAC_guess_probabilities(E_xy; n=n, e_c=e_c) #2^n x 2 x n array
        total_mut_info = total_mutual_info(condit_probabs, B_marginals_probabs; n=n)
        total_red_info = total_redundant_info(condit_probabs; n=n)
        #println("Total mutual info: ", total_mut_info)
        #println("Total redundant info: ", total_red_info)
        #println("Total info: ", total_mut_info - total_red_info)
        return total_mut_info - (total_red_info * (1/log(4)))
    end

    channel_capacity = 1 - H_bin((1+e_c)/2)
    return conditions.GeneralizedCorrelationCondition(;scenario=(n,n,2,2), functional=S_IC, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => channel_capacity))
end


"""
function limited_condit_probabs_PL0000(α, β; e_c)
    k_plus, k_minus = 1/2 + ((e_c/2) * (α + β)), 1/2 + ((e_c/2) * (α - β))
    limited_conditionals = Array{Float64}(undef, 2^2, 2, 2) #P(α_vec | B_i, i=β)
    limited_conditionals[1, 0+1, 1], limited_conditionals[1, 0+1, 2] = k_plus, k_plus
    limited_conditionals[2, 0+1, 1], limited_conditionals[2, 0+1, 2] = k_plus, 1-k_minus
    limited_conditionals[3, 0+1, 1], limited_conditionals[3, 0+1, 2] = 1-k_plus, k_minus
    limited_conditionals[4, 0+1, 1], limited_conditionals[4, 0+1, 2] = 1-k_plus, 1-k_plus
    
    limited_conditionals[1, 1+1, 1], limited_conditionals[1, 1+1, 2] = 1-limited_conditionals[1, 0+1, 1], 1-limited_conditionals[1, 0+1, 2]
    limited_conditionals[2, 1+1, 1], limited_conditionals[2, 1+1, 2] = 1-limited_conditionals[2, 0+1, 1], 1-limited_conditionals[2, 0+1, 2]
    limited_conditionals[3, 1+1, 1], limited_conditionals[3, 1+1, 2] = 1-limited_conditionals[3, 0+1, 1], 1-limited_conditionals[3, 0+1, 2]
    limited_conditionals[4, 1+1, 1], limited_conditionals[4, 1+1, 2] = 1-limited_conditionals[4, 0+1, 1], 1-limited_conditionals[4, 0+1, 2]
    
    B_marginals = fill(1/2, 2, 2)
    A_marginals = fill(1/(2^2), 2, 2)

    return limited_conditionals ./ 2, B_marginals
end

MaxMixedBox = nsboxes.reconstructFullJoint(UniformRandomBox((2, 2, 2, 2)))
PR(μ, ν, σ) = nsboxes.reconstructFullJoint(PRBoxesCHSH(;μ=μ, ν=ν, σ=σ))
CanonicalPR = PR(0, 0, 0)
PL(α, γ, β, λ) = nsboxes.reconstructFullJoint(LocalDeterministicBoxesCHSH(;α=α, γ=γ, β=β, λ=λ))

CHSH_score = games.canonical_CHSH_score
CHSHprime_score = games.CHSH_score_generator(1,-1,1,1; batched=false)


function Compute_Slice_Coeff(P1::Array{Float64,4}, P2::Array{Float64,4}, P3::Array{Float64,4}, CHSHprime_score_val::Real, CHSH_score_val::Real) # P1, P2, P3 are 2x2x2x2 tensors
    A = [CHSHprime_score(P1) CHSHprime_score(P2) CHSHprime_score(P3);
           CHSH_score(P1)     CHSH_score(P2)     CHSH_score(P3);
                1                    1                1           ]
    b = [CHSHprime_score_val, CHSH_score_val, 1]
    return A INSERT_BACKSLASH b # Equiv. to np.linalg.solve(A, b)
end

function Limited_RedundantInfo_IC_Bound(;e_c=0.001)

    n=2
    function S_IC(nsjoint::Array{Float64, 4})
        @assert (size(nsjoint,1),size(nsjoint,2)) == (n, n)
        @assert (size(nsjoint,3),size(nsjoint,4)) == (2, 2)

        E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator

        #condit_probabs, B_marginals_probabs = condit_and_B_marginal_RAC_guess_probabilities(E_xy; n=n, e_c=e_c) #2^n x 2 x n array
        alpha_val, _, beta_val = Compute_Slice_Coeff(CanonicalPR, MaxMixedBox, PL(0,0,0,0),CHSHprime_score(nsjoint), CHSH_score(nsjoint))
        condit_probabs, B_marginals_probabs = limited_condit_probabs_PL0000(alpha_val, beta_val; e_c=e_c)
         

        total_mut_info = total_mutual_info(condit_probabs, B_marginals_probabs; n=n)
        total_red_info = total_redundant_info(condit_probabs; n=n)
        println("Total mutual info: ", total_mut_info)
        println("Total redundant info: ", total_red_info)
        println("Total info: ", total_mut_info - total_red_info)
        return total_mut_info - (total_red_info * (1/log(4)))
    end

    channel_capacity = 1 - H_bin((1+e_c)/2)
    return conditions.GeneralizedCorrelationCondition(;scenario=(n,n,2,2), functional=S_IC, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => channel_capacity))
end


# ------------------------------------ #
# The following are pseudo-bounds in the sense that they are not worked out and very inefficient
# ------------------------------------ #

function Jain_Raw_IC_Bound(;n=2, e_c=0.01)
    
    function IC_ineq_statement(nsjoint::Array{Float64, 4})
        @assert (size(nsjoint,1),size(nsjoint,2)) == (n, n)
        @assert (size(nsjoint,3),size(nsjoint,4)) == (2, 2)
        @assert -1 <= e_c <= 1
        
        channel_capacity = 1 - H_bin((1+e_c)/2)
        succes_prob = IC_Success_P_GeneralVanDam(nsjoint; e_c=e_c)
        return sum(1-H_bin(p) for p in succes_prob) - channel_capacity
    end

    return conditions.GeneralizedCorrelationCondition(;scenario=(n,n,2,2), functional=IC_ineq_statement, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 0))
end




function Erasure_IC_Success_P_GeneralVanDam(nsjoint::Array{Float64,4}; e_η::Real, return_E_xy::Bool=false)
    n = size(nsjoint,3) #Number of inputs
    E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
    
    Psuccess = []
    #push!(Psuccess, (1+e_η)/4 + ((1-e_η)/2)*((1/(2^n))*(2*E_xy[0+1,0+1] + sum(2^j * E_xy[j+1,0+1] for j in 1:n-1))))
    push!(Psuccess, ((1-e_η)/2)*((1/(2^n))*(2*E_xy[0+1,0+1] + sum(2^j * E_xy[j+1,0+1] for j in 1:n-1))))
    
    for i in 1:n-1 #1-indexing shift
        #push!(Psuccess, (1+e_η)/4 + ((1-e_η)/2)*((1/(2^n))*(2*E_xy[0+1,i+1] + sum((-1)^(j==(n-i)) * 2^j * E_xy[j+1,i+1] for j in 1:n-i))))
        push!(Psuccess, ((1-e_η)/2)*((1/(2^n))*(2*E_xy[0+1,i+1] + sum((-1)^(j==(n-i)) * 2^j * E_xy[j+1,i+1] for j in 1:n-i))))
    end

    if return_E_xy
        return Psuccess, E_xy
    else
        return Psuccess
    end
end

function Erasure_Raw_IC_Bound(;n=2, e_η=0.99)
    
    function IC_ineq_statement(nsjoint::Array{Float64, 4}) #::nsboxes.NSBox
        @assert (size(nsjoint,1),size(nsjoint,2)) == (n, n)
        @assert (size(nsjoint,3),size(nsjoint,4)) == (2, 2)
        @assert -1 <= e_η <= 1 
        channel_capacity = (1-e_η)/2
        succes_prob = Erasure_IC_Success_P_GeneralVanDam(nsjoint; e_η=e_η)
        return sum(1-H_bin(p) for p in succes_prob) - channel_capacity
    end

    return conditions.GeneralizedCorrelationCondition(;scenario=(n,n,2,2), functional=IC_ineq_statement, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 0))
end

function ZChannel_IC_Success_P_GeneralVanDam(nsjoint::Array{Float64,4}; e_c::Real, return_E_xy::Bool=false)
    n = size(nsjoint,3) #Number of inputs
    E_xy = ProbBellCorrelator(nsjoint) #  ProbBellCorrelator
    box_A_marg = sum(nsjoint, dims=2)[:, 1, :, 1]
    @assert size(box_A_marg) == (2, n)
    
    Psuccess = []
    f(α::Vector{Int}) = length(α)==2 ? mod(α[1]+α[2], 2) : n - 1 - sum(prod(mod(α[0+1]+α[l+1], 2) for l in 1:i) for i in 1:n-1)
    P_μ = (1/(2^n))*sum(sum(Int(f(collect(m_vec))==r)*box_A_marg[m_vec[1]+1, r+1] for r in 0:n-1) for m_vec in Iterators.product(fill(0:1, n)...))
    
    push!(Psuccess, 1/2 + ((1+e_c)*P_μ - e_c)*((1/(2^n))*(2*E_xy[0+1,0+1] + sum(2^j * E_xy[j+1,0+1] for j in 1:n-1))))
    for i in 1:n-1 #1-indexing shift
        push!(Psuccess, 1/2 + ((1+e_c)*P_μ - e_c)*((1/(2^n))*(2*E_xy[0+1,i+1] + sum((-1)^(j==(n-i)) * 2^j * E_xy[j+1,i+1] for j in 1:n-i))))
    end

    if return_E_xy
        return Psuccess, E_xy
    else
        return Psuccess
    end
end

function ZChannel_Raw_IC_Bound(;n=2, e_c=0.01)
    
    function IC_ineq_statement(nsjoint::Array{Float64, 4}) #::nsboxes.NSBox
        @assert (size(nsjoint,1),size(nsjoint,2)) == (n, n)
        @assert (size(nsjoint,3),size(nsjoint,4)) == (2, 2)
        @assert -1 <= e_c <= 1 "Success bias e_c must be in [0,1]"
        
        p_c = (1+e_c)/2
        channel_capacity = log2(1 + ((1-p_c)*p_c^(p_c/(1-p_c))))
        succes_prob = ZChannel_IC_Success_P_GeneralVanDam(nsjoint; e_c=e_c)
        return sum(1-H_bin(p) for p in succes_prob) - channel_capacity
    end

    return conditions.GeneralizedCorrelationCondition(;scenario=(n,n,2,2), functional=IC_ineq_statement, compare_operator=(<=), bound=missing, setBounds=Dict(:Q => 0))
end

"""