module sdp_conditions

    using LinearAlgebra
    ###using PythonCall
    using JuMP, MosekTools

    !isdefined(Main, :nsboxes) ? error("First import the nsboxes module") : nothing
    using ..nsboxes
    
    #----------------------------------------------------------------------------
    #### ------ Other type of conditions: SDP based tests - NPA hierarchy -------
    #----------------------------------------------------------------------------


    #QuantumNPA.jl based implementations:
    #!isdefined(Main, :QuantumNPA) ? error("For SDP conditions, the modified QuantumNPA.jl module must be available") : nothing
    #using ..QuantumNPA
    include("QuantumNPA/QuantumNPA.jl")

    function is_in_NPA(FullNSJoint::Array{Float64,4}; level::Int=3, verbose=false)
        Box = nsboxes.NSBox((2,2,2,2), FullNSJoint)
        PA = QuantumNPA.projector(1, 1:2, 1:2, full=true)
        PB = QuantumNPA.projector(2, 1:2, 1:2, full=true)

        constraints = [PA[1,1] - Box.marginals_vec_A[1]*QuantumNPA.Id,
                        PA[1,2] - Box.marginals_vec_A[2]*QuantumNPA.Id,
                        PB[1,1] - Box.marginals_vec_B[1]*QuantumNPA.Id,
                        PB[1,2] - Box.marginals_vec_B[2]*QuantumNPA.Id,
                        (PA[1,i]*PB[1,j] - Box.joints_mat[i,j]*QuantumNPA.Id for (i,j) in Iterators.product(1:2, 1:2))...,
                        ]
        
        #j_mo, j_vars = npa2jump(1e-4*QuantumNPA.Id, level; eq=constraints, solver=Mosek.Optimizer, return_vars=true)
        
        #if !verbose
        #    set_attribute(j_mo, "QUIET", true)
        #    set_attribute(j_mo, "INTPNT_CO_TOL_DFEAS", 1e-7)
        #end


        return QuantumNPA.npa_max(1*QuantumNPA.Id, level; eq=constraints, solver=Mosek.Optimizer) > 0.0
    end


    function is_asymp_in_NPA(FullNSJoint::Array{Float64,4}; level::Int, ϵ=eps(Float32), verbose=false)
        function npa_test(λ)
            PA = QuantumNPA.projector(1, 1:2, 1:2, full=true)
            PB = QuantumNPA.projector(2, 1:2, 1:2, full=true)
    
            Box = nsboxes.NSBox((2,2,2,2), FullNSJoint)
            
            eq_constraints =  [PA[1,1] - (λ*Box.marginals_vec_A[1] + (1-λ)*0.5)*QuantumNPA.Id,
                                PA[1,2] - (λ*Box.marginals_vec_A[2] + (1-λ)*0.5)*QuantumNPA.Id,
                                PB[1,1] - (λ*Box.marginals_vec_B[1] + (1-λ)*0.5)*QuantumNPA.Id,
                                PB[1,2] - (λ*Box.marginals_vec_B[2] + (1-λ)*0.5)*QuantumNPA.Id,
                                (PA[1,i]*PB[1,j] - (λ*Box.joints_mat[i,j] + (1-λ)*0.25)*QuantumNPA.Id for (i,j) in Iterators.product(1:2, 1:2))...,
                                ]
           
            return QuantumNPA.npa_max(λ*QuantumNPA.Id, level; eq=eq_constraints, solver=Mosek.Optimizer) >= λ - ϵ #>= 0.0
        end
        
        # Binary search on continuous variable λ
        λ_interval = (0.0, 1.0)
    
        #First check endpoints:
        #extreme_output = (test(λ_interval[1]), test(λ_interval[end]))
        #if extreme_output == (true, true)
        #    println("Both endpoints are feasible")
        #    return false
        #elseif extreme_output == (false, false)
        #    println("Both endpoints are infeasible")
        #    return false
        #end
        
        c_λ_window = (λ_interval[1], λ_interval[end])
        while c_λ_window[2] - c_λ_window[1] > ϵ
            c_λ = (c_λ_window[1] + c_λ_window[2])/2
            c_output = npa_test(c_λ)
            c_λ_window = c_output ? (c_λ, c_λ_window[2]) : (c_λ_window[1], c_λ)
        end
        
        return c_λ_window[2] == λ_interval[2] || c_λ_window[1] >= λ_interval[2] - ϵ
    end

    function randomization_distance_to_NPA(FullNSJoint::Array{Float64,4}; level::Int, ϵ::AbstractFloat=eps(Float32), verbose=false)
        """ !! Non-orthogonal !! distance to the NPA set. """
        function npa_test(λ)
            PA = QuantumNPA.projector(1, 1:2, 1:2, full=true)
            PB = QuantumNPA.projector(2, 1:2, 1:2, full=true)
    
            Box = nsboxes.NSBox((2,2,2,2), FullNSJoint)
            
            eq_constraints =  [PA[1,1] - (λ*Box.marginals_vec_A[1] + (1-λ)*0.5)*QuantumNPA.Id,
                                PA[1,2] - (λ*Box.marginals_vec_A[2] + (1-λ)*0.5)*QuantumNPA.Id,
                                PB[1,1] - (λ*Box.marginals_vec_B[1] + (1-λ)*0.5)*QuantumNPA.Id,
                                PB[1,2] - (λ*Box.marginals_vec_B[2] + (1-λ)*0.5)*QuantumNPA.Id,
                                (PA[1,i]*PB[1,j] - (λ*Box.joints_mat[i,j] + (1-λ)*0.25)*QuantumNPA.Id for (i,j) in Iterators.product(1:2, 1:2))...,
                                ]
           
            return QuantumNPA.npa_max(λ*QuantumNPA.Id, level; eq=eq_constraints, solver=Mosek.Optimizer) >= λ - ϵ #>= 0.0
        end
        
        # Binary search on continuous variable λ
        λ_interval = (0.0, 1.0)
    
        #First check endpoints:
        #extreme_output = (npa_test(λ_interval[1]), npa_test(λ_interval[end]))
        #if extreme_output[2] == true
            #println("The point is in the NPA set; There is no distance to the NPA set")
        #    return 0.0
        #end
        
        c_λ_window = (λ_interval[1], λ_interval[end])
        while c_λ_window[2] - c_λ_window[1] > ϵ
            c_λ = (c_λ_window[1] + c_λ_window[2])/2
            c_output = npa_test(c_λ)
            c_λ_window = c_output ? (c_λ, c_λ_window[2]) : (c_λ_window[1], c_λ)
        end
        
        if c_λ_window[2] == λ_interval[2]
            return 0.0
        else
            return λ_interval[2] - (c_λ_window[1] + c_λ_window[2])/2
        end
    end


    function randomization_distance_to_NPA_boundary(FullNSJoint::Array{Float64,4}; level::Int, ϵ::AbstractFloat=eps(Float32), verbose=false)
        function npa_test(λ, η)
            PA = QuantumNPA.projector(1, 1:2, 1:2, full=true)
            PB = QuantumNPA.projector(2, 1:2, 1:2, full=true)
    
            Box = nsboxes.NSBox((2,2,2,2), FullNSJoint)
            
            eq_constraints =  [(η*PA[1,1] + ((1-η)*0.5*QuantumNPA.Id)) - (λ*Box.marginals_vec_A[1] + (1-λ)*0.5)*QuantumNPA.Id,
                            (η*PA[1,2] + ((1-η)*0.5*QuantumNPA.Id)) - (λ*Box.marginals_vec_A[2] + (1-λ)*0.5)*QuantumNPA.Id,
                            (η*PB[1,1] + ((1-η)*0.5*QuantumNPA.Id)) - (λ*Box.marginals_vec_B[1] + (1-λ)*0.5)*QuantumNPA.Id,
                            (η*PB[1,2] + ((1-η)*0.5*QuantumNPA.Id)) - (λ*Box.marginals_vec_B[2] + (1-λ)*0.5)*QuantumNPA.Id,
                            (η*PA[1,i]*PB[1,j] + ((1-η)*0.25*QuantumNPA.Id) - (λ*Box.joints_mat[i,j] + (1-λ)*0.25)*QuantumNPA.Id for (i,j) in Iterators.product(1:2, 1:2))...,
                            ]
           
            return QuantumNPA.npa_max(λ*QuantumNPA.Id, level; eq=eq_constraints, solver=Mosek.Optimizer) >= λ - ϵ #>= 0.0
        end
    
        if is_in_NPA(FullNSJoint; level=level, verbose=verbose)
        
            # Binary search on continuous variable λ
            λ_interval = (0.0, 1.0)
            η_interval = (0.0, 1.0)
            
            c_λ_window = (λ_interval[1], λ_interval[end])
            c_η_window = (η_interval[1], η_interval[end])
            prev_λ, prev_η = missing, missing
    
            while c_λ_window[2] - c_λ_window[1] > ϵ && c_η_window[2] - c_η_window[1] > ϵ
                c_λ = (c_λ_window[1] + c_λ_window[2])/2
                c_η = (c_η_window[1] + c_η_window[2])/2
                
                if ismissing(prev_λ) || ismissing(prev_η)
                    c_inward_output, c_outward_output = npa_test(c_λ, c_η), npa_test(c_λ, c_η)
                else
                    c_inward_output, c_outward_output = npa_test(c_λ, prev_η), npa_test(prev_λ, c_η)
                end
                
                if c_outward_output && c_inward_output
                    c_λ_window = (c_λ, c_λ_window[2])
                    c_η_window = (c_η_window[1], c_η)
                elseif !c_outward_output && c_inward_output
                    c_λ_window = (c_λ_window[1], c_λ)
                    c_η_window = (c_η, c_η_window[2])
                elseif c_outward_output && !c_inward_output
                    c_λ_window = (c_λ, c_λ_window[2])
                    c_η_window = (c_η_window[1], c_η)
                elseif !c_outward_output && !c_inward_output
                    c_λ_window = (c_λ_window[1], c_λ)
                    c_η_window = (c_η, c_η_window[2]) 
                end
            end
            
            outer_dist = (λ_interval[2] - (c_λ_window[1] + c_λ_window[2])/2)
            inner_dist = (η_interval[2] - (c_η_window[1] + c_η_window[2])/2)
            
            #@show outer_dist, inner_dist
            return -(inner_dist - outer_dist)
    
        else
            return randomization_distance_to_NPA(FullNSJoint; level=level, ϵ=ϵ, verbose=verbose)
        end
    end


    #ncpol2sdpa.py based implementations:
    ###pyimport("ncpol2sdpa")

    ###pyiter = pyimport("itertools")
    ###n2s = pyimport("ncpol2sdpa")

    function is_in_pyNPA(p_obs::Array{Float64, 4}; level::Int=2, verbose=false)
        P = n2s.Probability([2, 2], [2, 2])
        sdp = n2s.SdpRelaxation(P.get_all_operators(), verbose=false,
                                parallel=true)

        constraints = [P([a, b], [x, y]) - p_obs[a+1, b+1, x+1, y+1]
                    for (x, y, a, b) in Iterators.product(0:1, 0:1, 0:1, 0:1)]
        
        sdp.get_relaxation(level,
                            momentequalities=constraints,
                            substitutions=P.substitutions)
        sdp.solve(solver="mosek")
        
        if pyconvert(String, sdp.status) == "optimal"
            return true
        else
            return false
        end
    end


    function is_asymp_in_pyNPA(p_obs::Array{Float64, 4}; level::Int=2, verbose=false)
        λ = n2s.generate_variables("λ", 1)
        P = n2s.Probability([2, 2], [2, 2])
        sdp = n2s.SdpRelaxation(pylist([P.get_all_operators()..., λ[0]]), verbose=false,
                                parallel=true)
        
        objective = λ[0]
        #P = (1-λ)*p_obs + λ*MaxMiedBox
        eq_constraints = [P([a, b], [x, y]) - ((1-λ[0])*p_obs[a+1, b+1, x+1, y+1] + λ[0]*(1/4))  #* MaxMixedBox[a+1, b+1, x+1, y+1])
                    for (x, y, a, b) in Iterators.product(0:1, 0:1, 0:1, 0:1)]
        
        
        sdp.get_relaxation(level,
                            objective=objective,
                            momentequalities=eq_constraints,
                            substitutions=P.substitutions)
                            
        sdp.solve(solver="mosek")
        
        verbose && println("Found primal value: ", pyconvert(Float64, sdp.primal))

        if pyconvert(String, sdp.status) == "optimal" && pyconvert(Float64,sdp.primal) < 1e-3 

            return true
        else
            return false
        end
    end

    function optimize_distance_to_pyNPA(p_obs::Array{Float64, 4}; level::Int, verbose=false, solver="mosek")
        """Distance here measured as total variation distance between the observed distribution and the closest distribution in the NPA hierarchy"""
        P = n2s.Probability([2, 2], [2, 2])
        X = reshape(pyconvert(Vector, n2s.generate_variables("X", prod(size(p_obs)), commutative=true)), size(p_obs)...)
        sdp_vars = pylist([P.get_all_operators()..., reshape(X,:)...])
        sdp = n2s.SdpRelaxation( sdp_vars, verbose=false,
                                parallel=true)

        objective = sum(X)  
        
        #P - p_obs <= Δ
        #-Δ <= P - p_obs
        ineq_constraints = []
        for (i,j,k,l) in Iterators.product(0:1, 0:1, 0:1, 0:1)
            push!(ineq_constraints, X[i+1,j+1,k+1,l+1] - (P([i,j], [k,l]) - p_obs[i+1,j+1,k+1,l+1]))
            push!(ineq_constraints, X[i+1,j+1,k+1,l+1] + (P([i,j], [k,l]) - p_obs[i+1,j+1,k+1,l+1]))
        end

        sdp.get_relaxation(level=level,
                        objective=objective,
                        momentinequalities=ineq_constraints,
                        substitutions=P.substitutions)

        sdp.solve(solver="mosek")
        
        if pyconvert(String, sdp.status) == "optimal"
            verbose && println(pyconvert(Float64,sdp.primal), " <= dist <= ", pyconvert(Float64,sdp.dual))

            return sdp, Dict(:P => P, :X => X)
        else
            error("No optimal solution found")
        end
    end

    function min_distance_to_pyNPA(p_obs::Array{Float64, 4}; level::Int=2, verbose=false, solver="mosek")
        solved_sdp, _ = optimize_distance_to_pyNPA(p_obs; level=level, verbose=verbose, solver=solver)
        
        return pyconvert(Float64, solved_sdp.primal)
    end

    function nearest_pyNPA_point(p_obs::Array{Float64, 4}; level::Int=2, verbose=false, solver="mosek")
        solved_sdp, sdp_vars = optimize_distance_to_pyNPA(p_obs; level=level, verbose=verbose, solver=solver)
        
        P_sol = Array{Float64}(undef, 2, 2, 2, 2)
        for (x, y, a, b) in Iterators.product(0:1, 0:1, 0:1, 0:1)
            P_sol[a+1, b+1, x+1, y+1] = pyconvert(Float64, solved_sdp[sdp_vars[:P]([a, b], [x, y])])
        end
        return P_sol
    end



end
