!isdefined(Main, :nsboxes) ? (include("nsboxes.jl"); using .nsboxes) : nothing
!isdefined(Main, :conditions) ? (include("conditions.jl"); using .conditions) : nothing

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


# The following are pseudo-bounds in the sense that they are not worked out and very inefficient

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

