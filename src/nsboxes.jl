#CTRL+I = Copilot

module nsboxes
    using Symbolics

    #export NSBox, NSBoxFamily, NSBoxMixture, reconstructFullJoint

    global_eps_tol = 1e-10
    Base.nameof(x::Symbolics.Arr) = Symbol(split(string(x), "[")[1]) #Symbolics.jl arrays are named as "a[1]" etc. This function returns the name of the array as a symbol

    function method_argnames(m::Method)
        """Returns the argument names of a method as a vector of symbols
        """
        argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), m.slot_syms)
        #kwargnames = Base.kwarg_decl(m)
    
        if isempty(argnames)
            return argnames
        end
        #return cat((argnames[1:m.nargs])[2:end], kwargnames, dims=1)
        return (argnames[1:m.nargs])[2:end]
    end
    
    method_argnames(f::Function) = method_argnames(last(collect(methods(f))))
    Base.kwarg_decl(f::Function) = Base.kwarg_decl(last(collect(methods(f))))
    
    tuplejoin(x, y) = (x..., y...)


    mutable struct NSBox
        """Bipartite (family of) No-Signaling Box in (M_A, M_B, m_A, m_B)-scenario 
        Described by P(a,b|x,y) and stored in Collins-Gisin representation: (iteration order: inputs, outputs)
        If NSBox describes family of NS Boxes, then family_parameters identify used symbolic variables (symbolics.jl), otherwise family_parameters = Missing

        CAUTION: The following table is in row-first order, like the array indices (A along rows, B along columns)
                               || P(b = 0|y = 0)     | ... | P(b = m_B - 1|y = 0)     | P(b = 0|y = 1)     | ... | P(b = m_B - 1|y = M_B)     ||
        ========================================================================================================================================
        P(a = 0|x = 0)         || P(0,0|0,0)         | ... | P(0,m_B - 1|0,0)         | P(0,0|0,1)         | ... | P(0,m_B - 1|0,M_B)         ||
        ...                    || ...                | ... | ...                      | ...                | ... | ...                        ||
        P(a = m_A - 1|x = 0)   || P(m_A - 1,0|0,0)   | ... | P(m_A - 1,m_B - 1|0,0)   | P(m_A - 1,0|0,1)   | ... | P(m_A - 1,m_B - 1|0,M_B)   ||
        ----------------------------------------------------------------------------------------------------------------------------------------
        P(a = 0|x = 1)         || P(0,0|1,0)         | ... | P(0,m_B - 1|1,0)         | P(0,0|1,1)         | ... | P(0,m_B - 1|1,M_B)         ||
        ...                    || ...                | ... | ...                      | ...                | ... | ...                        ||
        ----------------------------------------------------------------------------------------------------------------------------------------
        ...                    || ...                | ... | ...                      | ...                | ... | ...                        ||
        P(a = m_A - 1|x = M_A) || P(m_A - 1,0|M_A,0) | ... | P(m_A - 1,m_B - 1|M_A,0) | P(m_A - 1,0|M_A,1) | ... | P(m_A - 1,m_B - 1|M_A,M_B) ||
        ========================================================================================================================================
        
        No-Signaling: P(a, m_B|x,y) = P(a | x(,y)) - [ P(a,0|x,y) + ... + P(a,m_B-1|x,y) ]
        + Normalization constraints: P(0,0|x,y) + ... + P(0, m_B|x,y) + P(1,0|x,y) + ... + P(1, m_B|x,y) + ... + P(m_A,m_B|x,y) = 1 
        -> Number of independent distribution parameters: M_A M_B (m_A - 1) (m_B - 1) + M_A (m_A - 1) + M_B (m_B - 1) (vs. M_A M_B m_A m_B - 1 for general bipartite behaviors)
        
        Attributes:
        ===============
        scenario::Tuple = Number of inputs M_A, M_B and outputs m_A, m_B for A and B respectively
        marginals_vec_A::Vector = Marginals for M_A of the inputs and of the (m_A-1) outputs of A
        marginals_vec_B::Vector = Marginals for M_B of the inputs and of the (m_B-1) outputs of B
        joints_mat::Matrix = Joint probabilities for M_A x M_B of the joint inputs (x,y) and (m_A-1) x (m_B-1) of the joint outputs of A and B, so dimensions are M_A(m_A-1) x M_B(m_B-1)
        """
        
        scenario::NTuple{4, <:Int}
        marginals_vec_A::Vector{Float64}
        marginals_vec_B::Vector{Float64}
        joints_mat::Matrix{Float64}
        #In the following constructor, all keyowrds are set after the semicolon to enforce exact keyword specification
        NSBox(;scenario,marginals_vec_A, marginals_vec_B, joints_mat) = new(scenario, marginals_vec_A, marginals_vec_B, joints_mat)
    end

    function NSBox(scenario::NTuple{4, <:Int}, marginals_vec_A::Vector{Float64}, marginals_vec_B::Vector{Float64}, joints_mat::Matrix{Float64})
        """Construct No-Signaling Box (::NSBox) in (M_A, M_B, m_A, m_B)-scenario from a Collins-Gisin representation (see NSBox definition)
        """
        @assert length(marginals_vec_A) == scenario[1] * (scenario[3] - 1) "Marginals of A should have length M_A * (m_A - 1)"
        @assert length(marginals_vec_B) == scenario[2] * (scenario[4] - 1) "Marginals of B should have length M_B * (m_B - 1)"
        @assert size(joints_mat) == (scenario[1]*(scenario[3]-1), scenario[2]*(scenario[4]-1)) "Joint probabilities should have dimensions M_A * (m_A - 1) x M_B * (m_B - 1)"
        
        return NSBox(scenario=scenario, marginals_vec_A=marginals_vec_A, marginals_vec_B=marginals_vec_B, joints_mat=joints_mat)
    end

    function NSBox(scenario::NTuple{4, <:Int}, full_joint::Array{Float64,4}; unsafe=false)
        """ Alternative constructor for NSBox (more efficient representation) from a full joint distribution P(a,b|x,y) as a 4D array
            
            The full joint P(a,b|x,y) should be given as a 4D array with indices [a,b,x,y] and dimensions (m_A, m_B, M_A, M_B)
        """

        #Check that scenario matches the dimensions of the full joint distribution:
        #@show (scenario[3], scenario[4], scenario[1], scenario[2]) == size(full_joint)
        !unsafe && (@assert all(size(full_joint) == (scenario[3], scenario[4], scenario[1], scenario[2])) "Scenario does not match dimensions of full joint distribution")

        #Check that both normalization and no-signaling constraints are fulfilled for the input for all (x,y) pairs:
        #For no-signaling on maringals of B, we compare whhether the marginals P(b|x,y) are equal for all y for any x. 
        full_A_marginals = sum(full_joint, dims=2)
        full_B_marginals = sum(full_joint, dims=1)
        
        if !unsafe
            @assert all(abs.(sum(full_A_marginals, dims=(1,2)) .- 1.0) .< global_eps_tol) "Joint probability not properly normalized by $(maximum(abs.(sum(full_A_marginals, dims=(1,2)) .- 1.0)))" #Normalization constraint on P(a,b|x,y) for all (x,y) pairs
            @assert all(abs.(full_A_marginals .-full_A_marginals[:,:,:,1:1]) .< global_eps_tol) "Marginals of A not independent of B; Signaling from B to A; Deviation by $(maximum(abs.(full_A_marginals .-full_A_marginals[:,:,:,1:1])))" #No-signaling constraints for marginals of A
            @assert all(abs.(full_B_marginals .-full_B_marginals[:,:,1:1,:]) .< global_eps_tol) "Marginals of B not independent of A; Signaling from A to B; Deviation by $(maximum(abs.(full_B_marginals .-full_B_marginals[:,:,1:1,:])))"#No-signaling constraints for marginals of B
            #@assert all(abs.(sum(full_A_marginals, dims=(1,2)) .- sum(full_B_marginals, dims=(1,2))) .< global_eps_tol) #Dummy check that normalization check is valid. TODO: Remove this line
        end

        reduced_joints_mat = reshape(permutedims(full_joint[1:end-1, 1:end-1, :, :], (1,3,2,4)), (scenario[1]*(scenario[3]-1), scenario[2]*(scenario[4]-1)))
        marginals_vec_A = reshape(full_A_marginals[1:end-1, :, :, 1], (scenario[1]*(scenario[3]-1))) #Note, 2nd (b) dimension should have length 1 at this point and y=1 could have been anything else than 1 (N.S. constraints A)
        marginals_vec_B = reshape(full_B_marginals[:, 1:end-1, 1, :], (scenario[2]*(scenario[4]-1))) #Note, 1st (a) dimension should have length 1 at this point and x=1 could have been anything else than 1 (N.S. constraints B)

        return NSBox(scenario=scenario, marginals_vec_A=marginals_vec_A, marginals_vec_B=marginals_vec_B, joints_mat=reduced_joints_mat)
    end 

    function Base.show(io::IO, nb::NSBox)
        rounded_A_marginals = string.(round.(nb.marginals_vec_A, digits=4))
        rounded_A_marginals = [elem*repeat("0", maximum(length.(rounded_A_marginals))-length(elem)) for elem in rounded_A_marginals]
        
        rounded_B_marginals = string.(round.(nb.marginals_vec_B, digits=4))
        rounded_B_marginals = [elem*repeat("0", maximum(length.(rounded_B_marginals))-length(elem)) for elem in rounded_B_marginals]

        rounded_joints_mat = string.(round.(nb.joints_mat, digits=4))

        first_column_vec = rounded_A_marginals
        remain_columns_vec = [" || " * join([elem*repeat("0", maximum(length.(rounded_B_marginals))-length(elem)) for elem in rounded_joints_mat[row_i, :]], " | ") * " ||" for row_i in 1:length(nb.marginals_vec_A)]
        
        first_line = repeat(" ", max(1, maximum(length.(first_column_vec)) + 1))*"|| " * join(rounded_B_marginals, " | ") * " ||"
        println(io, first_line)
        println(io, repeat("=", length(first_line)))
        for row_i in 1:length(nb.marginals_vec_A)
            println(io, first_column_vec[row_i] * remain_columns_vec[row_i])
        end
    end

    function Base.getindex(nsbox::NSBox, a::Int, b::Int, x::Int, y::Int)
        #@assert all((a,b,x,y) .!= 0) "Give 1-indexed indices, like for Julia arrays"
        reshaped_joints_mat = permutedims(reshape(nsbox.joints_mat, (nsbox.scenario[3] - 1,nsbox.scenario[1],nsbox.scenario[4] - 1, nsbox.scenario[2] )), (1,3,2,4)) 
        if a != nsbox.scenario[3] && b != nsbox.scenario[4]
            return reshaped_joints_mat[a,b,x,y]
        elseif a == nsbox.scenario[3] && b != nsbox.scenario[4]
            return (reshape(nsbox.marginals_vec_B, (1, nsbox.scenario[3] - 1, 1, nsbox.scenario[1]))[:,:,x,y] .- sum(reshaped_joints_mat[:,:,x,y], dims=1))[1,b] 
        elseif a != nsbox.scenario[3] && b == nsbox.scenario[4]
            return (reshape(nsbox.marginals_vec_A, (nsbox.scenario[3] - 1, 1, nsbox.scenario[1], 1))[:,:,x,y] .- sum(reshaped_joints_mat[:,:,x,y], dims=2))[a,1] 
        else #a == nsbox.scenario[3] && b == nsbox.scenario[4]
            Pa_at_last_b = reshape(nsbox.marginals_vec_A, (nsbox.scenario[3] - 1, 1, nsbox.scenario[1], 1))[:,:,x,y] .- sum(reshaped_joints_mat[:,:,x,y], dims=2) #P(a, b=d|x=x,y=y)
            Pb_at_last_a = reshape(nsbox.marginals_vec_B, (1, nsbox.scenario[3] - 1, 1, nsbox.scenario[1]))[:,:,x,y] .- sum(reshaped_joints_mat[:,:,x,y], dims=1) #P(a=d, b|x=x,y=y)
            return 1.0 .- (sum(reshaped_joints_mat[:,:,x,y], dims=(1,2)) .+ sum(Pb_at_last_a, dims=2) .+ sum(Pa_at_last_b, dims=1))
        end
    end

    function reconstructFullJoint(nsbox::NSBox)
        """Returns the full joint distribution P(a,b | x,y) as a 4D array from a NSBox
        Caution: The scenario specification (M_A, M_B, m_A, m_B) is the reverse order of the indices of the returned array (a,b,x,y)!

        Arguments:
        ==========
        nsbox::NSBox = NSBox

        Returns:
        =========
        Array{Float64,4} = 4D array of the full joint distribution P(a,b | x,y) with indices [a,b,x,y] and dimensions (m_A, m_B, M_A, M_B)
        """
        #full_joint = zeros(nsbox.scenario[3], nsbox.scenario[4], nsbox.scenario[1], nsbox.scenario[2])
        
        #----Test area -----
        """
        test_scene = (3,3,4,4)
        k1, k2 = [(a,x) for x in 1:test_scene[1] for a in 1:test_scene[3]-1], [(b,y) for y in 1:test_scene[2] for b in 1:test_scene[4]-1]
        partial_joint_test = [(k[1], l[1], k[2], l[2]) for k in k1, l in k2] #outer loop is first index, inner loop is second index; major-index (the last) is changing faster
        marginals_A_test = [(a,x) for x in 1:test_scene[1] for a in 1:test_scene[3]-1]
        marginals_B_test = [(b,y) for y in 1:test_scene[2] for b in 1:test_scene[4]-1]

        nsbox = NSBox(scenario=test_scene, 
                            marginals_vec_A=marginals_A_test,
                            marginals_vec_B=marginals_B_test,
                            joints_mat=partial_joint_test)
        #display(reshape(partial_joint_test, :))
        #display(partial_joint_test)
        display(marginals_A_test)
        
        full_joint = Array{Tuple}(undef, nsbox.scenario[3], nsbox.scenario[4], nsbox.scenario[1], nsbox.scenario[2])
        
        """
        # ------------------
        full_joint = Array{Union{Float64}}(undef, nsbox.scenario[3], nsbox.scenario[4], nsbox.scenario[1], nsbox.scenario[2])
        
        @assert length(nsbox.marginals_vec_A) == nsbox.scenario[1] * (nsbox.scenario[3]-1)
        @assert length(nsbox.marginals_vec_B) == nsbox.scenario[2] * (nsbox.scenario[4]-1)
        @assert size(nsbox.joints_mat) == (nsbox.scenario[1]*(nsbox.scenario[3]-1), nsbox.scenario[2]*(nsbox.scenario[4]-1))

        #First we copy the bulk part one-to-one over:
        full_joint[1:size(full_joint)[1] - 1, 1:size(full_joint)[2] - 1, 1:size(full_joint)[3], 1:size(full_joint)[4]] = permutedims(reshape(nsbox.joints_mat, (size(full_joint)[1] - 1,size(full_joint)[3],size(full_joint)[2] - 1, size(full_joint)[4] )), (1,3,2,4)) 
        #Above reshape results in indices [a,x,b,y] because of their decreasing frequency with respect to the column-based flattened vector of the 2D matrix.

        #Then we apply the No-Signaling constraints to add the joint probabilities for a=m_a and b=m_b (for all x,y pairs), except P(a=m_a, b=m_b|x,y).
        expanded_A_marginals = reshape(nsbox.marginals_vec_A, (size(full_joint)[1] - 1, 1, size(full_joint)[3],1))
        full_joint[1:end-1, end:end, :, :] = expanded_A_marginals .- sum(full_joint[1:end-1, 1:end-1, :, :], dims=2) #sum over all outputs of B

        expanded_B_marginals = reshape(nsbox.marginals_vec_B, (1, size(full_joint)[2] - 1, 1, size(full_joint)[4]))
        full_joint[end:end, 1:end-1, :, :] = expanded_B_marginals .- sum(full_joint[1:end-1, 1:end-1, :, :], dims=1) #sum over all outputs of A

        #Finally we apply the normalization of P(a,b|x,y) for each (x,y) pair, which adds all the P(a=m_a, b=m_b|x,y).
        full_joint[end:end, end:end, :, :] = 1.0 .- (sum(full_joint[1:end-1, 1:end-1, :, :], dims=(1,2)) .+ sum(full_joint[end:end, 1:end-1, :, :], dims=2) .+ sum(full_joint[1:end-1, end:end, :, :], dims=(1,2)))
        
        #Now P(a,b|x,y) is fully reconstructed!

        #Check at which indices undefined elements are left:
        #display(full_joint) 
        
        return full_joint
    end

    function Base.:*(coeff::Real, nsbox::NSBox)
        """Multiplication of a NSBox with a scalar (Probabilistic application of a NSBox)
        """
        
        @assert 0.0 - 100*eps() <= coeff <= 1.0 + 100*eps() "NSBoxes can only be multiplied by valid probabilities (0 <= coeff <= 1)"
        return nsbox=>coeff
    end

    Base.:*(nsbox::NSBox, coeff::Real) = coeff * nsbox

    function Base.:+(ProbabilityNSBoxPair_1::Pair{NSBox, <:Real}, ProbabilityNSBoxPair_2::Pair{NSBox, <:Real})
        """Mixing of two NSBoxes with asscoiated probabilities
        """
        (nsbox_1, coeff_1) = ProbabilityNSBoxPair_1
        (nsbox_2, coeff_2) = ProbabilityNSBoxPair_2

        #Check that both NSBoxes have the same scenario:
        @assert nsbox_1.scenario == nsbox_2.scenario "NSBoxes can only be mixed if they are in the same scenario"
        @assert abs(coeff_1 + coeff_2 - 1.0) < global_eps_tol

        mixed_marginals_vec_A = coeff_1 * nsbox_1.marginals_vec_A + coeff_2 * nsbox_2.marginals_vec_A
        mixed_marginals_vec_B = coeff_1 * nsbox_1.marginals_vec_B + coeff_2 * nsbox_2.marginals_vec_B
        mixed_joints_mat = coeff_1 * nsbox_1.joints_mat + coeff_2 * nsbox_2.joints_mat
        return NSBox(scenario=nsbox_1.scenario, marginals_vec_A=mixed_marginals_vec_A, marginals_vec_B=mixed_marginals_vec_B, joints_mat=mixed_joints_mat)
    end

    function Base.:+(a::Pair{NSBox,<:Real}, b::Pair{NSBox,<:Real}, args::Pair{NSBox,<:Real}...)

        #Check that both NSBoxes have the same scenario:
        @assert a.first.scenario == b.first.scenario "NSBoxes can only be mixed if they are in the same scenario"
        for arg in args
            arg.first.scenario == a.first.scenario
        end
        @assert abs(a.second + b.second + sum(arg.second for arg in args) - 1.0) < global_eps_tol "Deviation from 1.0: $(a.second + b.second + sum(arg.second for arg in args))"
        
        mixed_marginals_vec_A = a.second * a.first.marginals_vec_A + b.second * b.first.marginals_vec_A + reduce(+, arg.second * arg.first.marginals_vec_A for arg in args)
        mixed_marginals_vec_B = a.second * a.first.marginals_vec_B + b.second * b.first.marginals_vec_B + reduce(+, arg.second * arg.first.marginals_vec_B for arg in args)
        mixed_joints_mat = a.second * a.first.joints_mat + b.second * b.first.joints_mat + reduce(+, arg.second * arg.first.joints_mat for arg in args)
        return NSBox(scenario=a.first.scenario, marginals_vec_A=mixed_marginals_vec_A, marginals_vec_B=mixed_marginals_vec_B, joints_mat=mixed_joints_mat)
    end


    #-------------------------------------------------------
    #NSBox families:
    mutable struct NSBoxFamily
        """Family of (parameterized) NSBoxes
        """
        senario::NTuple{4, <:Int} #Number of inputs M_A, M_B and outputs m_A, m_B for A and B respectively
        parameters::Tuple{Vararg{<:Union{Num, Symbolics.Arr}}} #Vector of Symbolics.jl variables representing the parameters of the family of NSBoxes
        generator::Function #Function that takes a set of parameters and returns a NSBox instance
        function NSBoxFamily(;scenario::NTuple{4, <:Int}, parameters::Tuple{Vararg{<:Union{Num, Symbolics.Arr}}}, generator::Union{Method, Function})
            
            @assert length(method_argnames(generator))==0 "Generator function of NSBoxFamily should only have keyword arguments"
            @assert length(Base.kwarg_decl(generator))!=0 "Generator function of NSBoxFamily should have keyword arguments that specify the parameters of the NSBoxFamily"
            for param in parameters
                @assert nameof(param) in Base.kwarg_decl(generator) "Generator function of NSBoxFamily misses keyword argument for parameter $(nameof(param))"
            end

            safe_generator(;kwargs...)::NSBox = generator(;kwargs...) # Safety measure: If the generator doesn't give a NSBox, this will throw an error later on
            
            new(scenario, parameters, safe_generator)
        end
    end

    NSBoxFamily(scenario::NTuple{4, <:Int}, parameters::Tuple{Vararg{<:Union{Num, Symbolics.Arr}}}, generator::Union{Method, Function}) = NSBoxFamily(scenario=scenario, parameters=parameters, generator=generator)

    #function NSBoxFamily(scenario::NTuple{4, <:Int}, generator::Function)
    #    parameters = tuple(method_argnames(generator)...)
    #    return NSBoxFamily(scenario=scenario, parameters=parameters, generator=generator)
    #end
    
    function Base.:*(coeff::Real, nsbox_fam::NSBoxFamily)
        """Multiplication of a NSBoxFamily with a scalar (Probabilistic application of a NSBoxFamily)
        """
        @assert 0.0 <= coeff <= 1.0 "NSBoxFamilies can only be multiplied by valid probabilities (0 <= coeff <= 1)"
        return nsbox_fam=>coeff
    end

    Base.:*(nsbox_fam::NSBoxFamily, coeff::Real) = coeff * nsbox_fam

    #function Base.:+(a::Pair{<:Union{NSBox, NSBoxFamily},<:Real}, b::Pair{<:Union{NSBox, NSBoxFamily},<:Real}, args::Pair{<:Union{NSBox, NSBoxFamily},<:Real}...)
    #    @assert abs(a.second + b.second + sum(arg.second for arg in args) - 1.0) < global_eps_tol
    #    c = a.second + b.second
    #    return abs(c) > 10^(-10) ? Base.:+(c*((a.second/c)*a.first + (b.second/c)*b.first), args...) : (length(args) < 2 ? args[1].first : Base.:+(args...))  #Coefficient rescaling to preserve normalization and pair-type in mixture components
    #end

    #-------------------------------------------------------

    mutable struct NSBoxMixture
        """Convex mixture of NSBoxes and NSBoxFamilies with associated probabilities
        """
        scenario::NTuple{4, <:Int} #Number of inputs M_A, M_B and outputs m_A, m_B for A and B respectively
        parameters::Tuple{Vararg{<:Union{Num, Symbolics.Arr}}}  #Vector of symbols representing the parameters of some family of NSBoxes in the mixture
        nsboxes::Vector{Union{NSBox, NSBoxFamily}} #Vector of NSBoxes and NSBoxFamilies representing the mixture
        coefficients::Vector{Float64} #Vector of coefficients for the NSBoxes and NSBoxFamilies
        function NSBoxMixture(;scenario::NTuple{4, <:Int}, parameters::Tuple{Vararg{<:Union{Num, Symbolics.Arr}}}, nsboxes::Tuple{Union{NSBox, NSBoxFamily}}, coefficients::Vector{Float64})
            @assert length(nsboxes) == length(coefficients) "NSBoxes and NSBoxFamilies can only be mixed if they have the same number of elements"
            @assert all([nsbox.scenario == scenario for nsbox in nsboxes]) "NSBoxes and NSBoxFamilies can only be mixed if they are in the same scenario"
            @assert all([0.0 <= coeff <= 1.0 for coeff in coefficients]) "NSBoxes and NSBoxFamilies can only be mixed with valid probabilities (0 <= coeff <= 1)"
            @assert abs(sum(coefficients) - 1.0) < 1e-10 "NSBoxes and NSBoxFamilies can only be mixed with valid probabilities (0 <= coeff <= 1)"
            new(scenario, parameters, nsboxes, coefficients)
        end
    end

    function NSBoxMixture(nsboxes::Tuple{Vararg{<:Union{NSBox, NSBoxFamily}}}, coefficients::Vector{Float64})
        common_parameters = []
        for nsbox in nsboxes
            if isa(nsbox, NSBoxFamily)
                common_parameters = push!(common_parameters, nsbox.parameters...)
            end
        end
        return NSBoxMixture(scenario=nsboxes[1].scenario, parameters= isempty(common_parameters) ? tuple() : tuple(unique(common_parameters)), nsboxes=nsboxes, coefficients=coefficients)
    end	

    function NSBoxMixture(nsbox::NSBox)
        return NSBoxMixture(scenario=nsbox.scenario, parameters=tuple(), nsboxes=(nsbox,), coefficients=[1.0,])
    end

    function NSBoxMixture(nsbox_fam::NSBoxFamily)
        return NSBoxMixture(scenario=nsbox_fam.scenario, parameters=nsbox_fam.parameters, nsboxes=(nsbox_fam,), coefficients=[1.0,])
    end

    function Base.:*(coeff::Real, nsbox_mixture::NSBoxMixture)
        """Multiplication of a NSBoxMixture with a scalar (Probabilistic application of a NSBoxMixture)
        """
        @assert 0.0 <= coeff <= 1.0 "NSBoxMixtures can only be multiplied by valid probabilities (0 <= coeff <= 1)"
        return nsbox_mixture=>coeff
    end

    Base.:*(nsbox_mixture::NSBoxMixture, coeff::Real) = coeff * nsbox_mixture

    function Base.:+(ProbabilityNSBoxMixturePair_1::Pair{NSBoxMixture, <:Real}, ProbabilityNSBoxMixturePair_2::Pair{NSBoxMixture, <:Real})
        """Mixing of two NSBoxMixtures with asscoiated probabilities
        """
        (nsbox_mixture_1, coeff_1) = ProbabilityNSBoxMixturePair_1
        (nsbox_mixture_2, coeff_2) = ProbabilityNSBoxMixturePair_2

        #Check that both NSBoxMixtures have the same scenario:
        @assert nsbox_mixture_1.scenario == nsbox_mixture_2.scenario "NSBoxMixtures can only be mixed if they are in the same scenario"

        eps_tol = 1e-10
        @assert (coeff_1 + coeff_2 - 1.0) < global_eps_tol "NSBoxMixtures can only be mixed with valid probabilities (0 <= coeff <= 1)"

        mixed_nsboxes = tuplejoin(nsbox_mixture_1.nsboxes..., nsbox_mixture_2.nsboxes...)
        mixed_coefficients = vcat(coeff_1 * nsbox_mixture_1.coefficients, coeff_2 * nsbox_mixture_2.coefficients)
        mixed_parameters = tuple(unique(tuplejoin(nsbox_mixture_1.parameters..., nsbox_mixture_2.parameters...)))
        return NSBoxMixture(scenario=nsbox_mixture_1.scenario, parameters=mixed_parameters, nsboxes=mixed_nsboxes, coefficients=mixed_coefficients)
    end
    
    function Base.:+(a::Pair{NSBoxFamily, <:Real}, b::Pair{NSBoxFamily,<:Real})
        """Mixing of two NSBoxFamilies with asscoiated probabilities
        """
        (nsbox_fam_1, coeff_1) = a
        (nsbox_fam_2, coeff_2) = b

        #Check that both NSBoxFamilies have the same scenario:
        @assert nsbox_fam_1.scenario == nsbox_fam_2.scenario "NSBoxFamilies can only be mixed if they are in the same scenario"

        @assert (coeff_1 + coeff_2 - 1.0) < global_eps_tol "NSBoxFamilies can only be mixed with valid probabilities (0 <= coeff <= 1)"

        mixed_nsboxes = tuplejoin((nsbox_fam_1,), (nsbox_fam_2,))
        mixed_coefficients = [coeff_1, coeff_2]
        mixed_parameters = tuple(unique(tuplejoin(nsbox_fam_1.parameters..., nsbox_fam_2.parameters...)))
        return NSBoxMixture(scenario=nsbox_fam_1.scenario, parameters=mixed_parameters, nsboxes=mixed_nsboxes, coefficients=mixed_coefficients)
    end

    function Base.:+(a::Pair{NSBoxFamily,<:Real}, b::Pair{NSBox, <:Real})
        """Mixing of a NSBoxFamily and a NSBox with asscoiated probabilities
        """
        (nsbox_fam, coeff_1) = a
        (nsbox, coeff_2) = b

        #Check that both NSBoxes have the same scenario:
        @assert nsbox_fam.scenario == nsbox.scenario "NSBoxFamilies and NSBoxes can only be mixed if they are in the same scenario"

        @assert (coeff_1 + coeff_2 - 1.0) < global_eps_tol "NSBoxFamilies and NSBoxes can only be mixed with valid probabilities (0 <= coeff <= 1)"

        mixed_nsboxes = tuplejoin((nsbox_fam,), (nsbox,))
        mixed_coefficients = [coeff_1, coeff_2]
        mixed_parameters = tuple(unique(tuplejoin(nsbox_fam.parameters..., tuple())))
        return NSBoxMixture(scenario=nsbox_fam.scenario, parameters=mixed_parameters, nsboxes=mixed_nsboxes, coefficients=mixed_coefficients)
    end

    Base.:+(a::Pair{NSBox, <:Real}, b::Pair{NSBoxFamily, <:Real}) = b + a

    Base.:+(a::Pair{NSBoxMixture, <:Real}, b::Pair{NSBox, <:Real}) = a + (b.second * NSBoxMixture(b.first))
    Base.:+(a::Pair{NSBox, <:Real}, b::Pair{NSBoxMixture, <:Real}) = (a.second * NSBoxMixture(a.first)) + b
    Base.:+(a::Pair{NSBoxMixture, <:Real}, b::Pair{NSBoxFamily, <:Real}) = a + (b.second * NSBoxMixture(b.first))
    Base.:+(a::Pair{NSBoxFamily, <:Real}, b::Pair{NSBoxMixture, <:Real}) = (a.second * NSBoxMixture(a.first)) + b




end #End of module nsboxes



