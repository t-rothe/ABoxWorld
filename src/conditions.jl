
module conditions
    #include("jl")
    #using .nsboxes

    #using Symbolics
    using LinearAlgebra
    using Distances

    !isdefined(Main, :nsboxes) ? error("First import the nsboxes module") : nothing
    using ..nsboxes

    #export GeneralizedCorrelationCondition, BellInequality, evaluate, check
    
    global_eps_tol = 1e-10 #Global tolerance for approximate comparisons
    #----------------- Conditions and criteria: -----------------

    struct GeneralizedCorrelationCondition
        """Generalized correlation condition with a specific (general) bound"""
        scenario::NTuple{4, <:Int} #Number of inputs M_A, M_B and outputs m_A, m_B for A and B respectively
        functional::Base.Callable #Function that takes a the FullJoint of a NSBox and returns a Float64
        compare_operator::Base.Callable #Should be one of ==, <, >, <=, >= operators
        bound::Union{<:Real, Bool, Missing} #(Generic/Fixed) Bound of the functional
        setBounds::Union{Dict{Symbol, <:Union{<:Real, Bool}}, Missing} #Dictionary of bounds for any common set, e.g. :L, :NS, :Q
        function GeneralizedCorrelationCondition(;scenario::NTuple{4, <:Int}, functional::Base.Callable, compare_operator::Base.Callable, bound::Union{<:Real, Bool, Missing}, setBounds::Union{Dict{Symbol, <:Union{<:Real, Bool}}, Missing})
            @assert length(scenario) == 4
            @assert compare_operator ∈ [==, <, >, <=, >=]
            ismissing(bound) && (@assert !ismissing(setBounds) && length(setBounds) > 0) #Either a fixed bound or setBounds must be given
            new(scenario, functional, compare_operator, bound, setBounds)#
        end
    end

    #GeneralizedCorrelationCondition(;scenario::NTuple{4, <:Int}, functional::Base.Callable, compare_operator::Base.Callable, bound::Union{Bool, <:Real}) = GeneralizedCorrelationCondition(;scenario=scenario, functional=functional, compare_operator=compare_operator, bound=bound, setBounds=missing)
    #GeneralizedCorrelationCondition(;scenario::NTuple{4, <:Int}, functional::Base.Callable, compare_operator::Base.Callable, setBounds::Dict{Symbol, <:Union{<:Real, Bool}}) = GeneralizedCorrelationCondition(;scenario=scenario, functional=functional, compare_operator=compare_operator, bound=missing, setBounds=setBounds)


    function BellInequality(scenario::NTuple{4, <:Int}, v::Array{Float64,4}, bound::Float64, setBounds::Union{Dict{Symbol,Float64}, Missing})
        """Constructs a BellInequality from a functional on a NSBox
        """
        return GeneralizedCorrelationCondition(scenario=scenario, functional=v->sum(v .* reconstructFullJoint(nsbox)), compare_operator=(<=), bound=bound, setBounds=setBounds)
    end

    
    #----------------- Utility functions: -----------------

    function is_strictly_greater_than(x, y; atol=global_eps_tol)
        x > y && !isapprox(x, y; atol=atol)
    end
    
    function is_strictly_less_than(x, y; atol=global_eps_tol)
        x < y && !isapprox(x, y; atol=atol)
    end

    function is_greater_than_or_approx_equal(x, y; atol=global_eps_tol)
        x >= y || isapprox(x, y; atol=atol)
    end
    
    function is_less_than_or_approx_equal(x, y; atol=global_eps_tol)
        x <= y || isapprox(x, Float64(y); atol=atol)
    end

    approx_compare_ops = Dict((==) => isapprox, (<) => is_strictly_less_than, (>) => is_strictly_greater_than, (<=) => is_less_than_or_approx_equal, (>=) => is_greater_than_or_approx_equal)

    function evaluate(condition::GeneralizedCorrelationCondition, nsjoint::Array{Float64,4})
        """Evaluates the condition's functional on a given FullJoint
        """
        return condition.functional(nsjoint)
    end

    function evaluate(condition::GeneralizedCorrelationCondition, nsbox::nsboxes.NSBox; unsafe=false)
        """Evaluates the condition's functional on a given NSBox
        """
        !unsafe && (@assert condition.scenario == nsbox.scenario)
        return condition.functional(nsboxes.reconstructFullJoint(nsbox))
    end

    function evaluate(condition::GeneralizedCorrelationCondition, nsbox_fam::nsboxes.NSBoxFamily)
        warn("""Evaluate(⋅) on NSBoxFamilies is highly unstable and requires implementation of an "unsafe" version of the generator function to allow processing symbolic variables (=keyword arguments)! \n Custom generators can feature a "unsafe=false" keyword argument to only leave safe-mode when really necessary. \n This is currently not even implemented for all the pre-defined generators! \n Prefer using evaluate(⋅) on NSBoxes whenever you can!""")
        kwarg_dict = Dict(nameof(k) => k for k in nsbox_fam.parameters)
        try
            return evaluate(condition, nsbox_fam.generator(;kwarg_dict...))
        catch err1
            return evaluate(condition, nsbox_fam.generator(;kwarg_dict..., unsafe=true))
        end   
    end

    #----

    function check(condition::GeneralizedCorrelationCondition, nsjoint::Array{Float64,4}, set::Symbol; unsafe=false)
        """Checks whether the condition is satisfied for a given FullJoint and specific set (if condition in is a hierachy rather than a fixed bound)
        """
        !unsafe && (@assert !ismissing(condition.setBounds) && set ∈ keys(condition.setBounds)) #Either a fixed bound or specific set within setBounds must be given
        return approx_compare_ops[condition.compare_operator](evaluate(condition, nsjoint), condition.setBounds[set]; atol=global_eps_tol)
    end
    
    function check(condition::GeneralizedCorrelationCondition, nsbox::nsboxes.NSBox, set::Symbol; unsafe=false)
        return check(condition, nsboxes.reconstructFullJoint(nsbox), set; unsafe=unsafe)
    end

    #----

    function check(condition::GeneralizedCorrelationCondition, nsjoint::Array{Float64,4}; unsafe=false)
        """Checks whether the condition is satisfied for a given FullJoint
        """
        !unsafe && (@assert !ismissing(condition.bound)) #Either a fixed bound or specific set within setBounds must be given
        return approx_compare_ops[condition.compare_operator](evaluate(condition, nsjoint), condition.bound; atol=global_eps_tol)
    end
    function check(condition::GeneralizedCorrelationCondition, nsbox::nsboxes.NSBox; unsafe=false)
        return check(condition, nsboxes.reconstructFullJoint(nsbox); unsafe=unsafe)
    end

end