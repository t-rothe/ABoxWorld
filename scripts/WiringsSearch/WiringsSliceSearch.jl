
include(scriptsdir("WiringsSearch", "wirings_search.jl"))


@with_kw struct WiringsSliceSearchConfig
    mode::Symbol; @assert mode in [:uniform, :greedy_lifting]
    box_search_space::Symbol; @assert box_search_space in [:mid_mid_point, :full_IC_Q_gap]
    Box1::Pair{String, Array{Float64,4}}
    Box2::Pair{String, Array{Float64,4}}
    Box3::Pair{String, Array{Float64,4}}
    primary_score::Function = CHSH_score
    secondary_score::Function 
    boundary_precision::Float64
    search_precision::Float64; @assert search_precision >= boundary_precision #Search can't be more precise than the computed boundaries
    precision::Float64
    max_wiring_order::Int
end

#Adapt savename() for WiringsSliceSearchConfig:
DrWatson.default_prefix(c::WiringsSliceSearchConfig) = string(c.mode)*"_WiringsSliceSearch_"*string(c.box_search_space)*"_"*c.Box1.first*"_"*c.Box2.first*"_"*c.Box3.first*"_MaxOrder_"*string(c.max_wiring_order)
DrWatson.allaccess(::WiringsSliceSearchConfig) = Tuple()


function Wirings_Slice_Search(config::WiringsSliceSearchConfig; verbose::Bool=false)

    results = Dict(string(key) => getfield(config, key) for key in fieldnames(WiringsSliceSearchConfig))

    P1, P2, P3 = config.Box1.second, config.Box2.second, config.Box3.second
    secondary_score = config.secondary_score

    # Draw the initial background triangle:
    x1, y1 = secondary_score(P1), CHSH_score(P1) #Stays CHSH because it's the coordinate system of the plot, also in IC
    x2, y2 = secondary_score(P2), CHSH_score(P2)
    x3, y3 = secondary_score(P3), CHSH_score(P3)
    ns_extremes = [(x1, y1), (x2, y2), (x3, y3)]
    results["ns_extremes"] = ns_extremes

    # Check if our coordinate system is good, i.e. if projected points aren't aligned (= allow 2D viz.):
    ( (x1 - x3)*(y2-y3)==(x2-x3)*(y1-y3) ) && error("We can't make a 2D plot from a 0D or 1D space of points.")


    # Compute the Quantum boundary:
    print2log("Computing Quantum boundary...")
    membership_level = 3
    c_secondary_score_val=min(x2,x3) #Initialize at x-axis origin
    c_CHSH_score_val=4 #Safely initialize at the maximum local value of CHSH
    results["quantum_secondary_scores"] = Float64[]
    results["quantum_primary_scores"] = Float64[]
    while c_secondary_score_val <= max(x2, x3)
        if is_in_NPA(c_secondary_score_val, c_CHSH_score_val, P1, P2, P3, secondary_score, membership_level) 
            push!(results["quantum_secondary_scores"], c_secondary_score_val)
            push!(results["quantum_primary_scores"], c_CHSH_score_val)
            c_secondary_score_val += config.boundary_precision
        else 
            if c_CHSH_score_val - config.boundary_precision > y2  #Still on the canvas of our viz.
                c_CHSH_score_val -= config.boundary_precision
            else
                c_CHSH_score_val=y2 #Take the canvas lower-boundary as the quantum boundary (even if it's not in the Q set) 
                c_secondary_score_val += config.boundary_precision # Go to the next point along x-axis / vertical line 
            end
        end
    end


    # Compute the IC boundary:
    print2log("Computing IC boundary...")
    algebraic_max_CHSH_score = 4.0
    min_secondary_score = 0.0
    c_secondary_score_val = min(x2,x3) #Reset x-axis pointer to smallest value on the x-axis
    init_CHSH_guess = 2.0
    c_CHSH_score_val = init_CHSH_guess
    #In this scenario, need to ensure that Quantum and IC boundaries are aligned
    results["unwired_IC_secondary_scores"] = copy(results["quantum_secondary_scores"]) #Float64[]
    results["unwired_IC_primary_scores"] = Union{Float64, Missing}[]
    #
    for c_secondary_score_val in results["unwired_IC_secondary_scores"] 
        while c_CHSH_score_val <= algebraic_max_CHSH_score - c_secondary_score_val + min_secondary_score #Check whether still within NS region; equiv. to whether there still is a pair of (G, CHSH) that can be in the NS region

            if is_NOT_in_IC(c_secondary_score_val, c_CHSH_score_val, P1, P2, P3, secondary_score)
                push!(results["unwired_IC_primary_scores"], c_CHSH_score_val)
                c_CHSH_score_val = (algebraic_max_CHSH_score - max(x2, x3))*(c_secondary_score_val-min(x2, x3))/(max(x2, x3)-min(x2, x3)) + init_CHSH_guess*(c_secondary_score_val-max(x2, x3))/(min(x2,x3)-max(x2, x3)) # Lagrange (linear?) interpolation on plot origin (0.75, 0.75) and (0.5, init_CHSH_guess) for x-point of next iteration; Set new initial y-value to 
                break # Success, so move on to next point along x-axis
                #That's because, Above CHSH guess does not work if boundary happens to be concave! (might guess to high -> too loose boundary)
            else
                if c_CHSH_score_val + config.boundary_precision < (algebraic_max_CHSH_score-c_secondary_score_val)
                    if c_secondary_score_val > 0.12 && c_CHSH_score_val > 3.5
                        @show c_CHSH_score_val
                    end
                    c_CHSH_score_val += config.boundary_precision #Stay at same x-point, but go up along y-axis
                else 
                    #Accept that we're at the boundary of the NS region and move on to the next x-point
                    push!(results["unwired_IC_primary_scores"], missing) #Add the NS boundary point to the list
                    #push!(results["unwired_IC_primary_scores"], algebraic_max_CHSH_score - c_secondary_score_val + min_secondary_score) #Add the NS boundary point to the list
                    c_CHSH_score_val = (algebraic_max_CHSH_score - max(x2, x3))*(c_secondary_score_val-min(x2, x3))/(max(x2, x3)-min(x2, x3)) + init_CHSH_guess*(c_secondary_score_val-max(x2, x3))/(min(x2,x3)-max(x2, x3)) # Lagrange interpolation; Same as in comment above
                    break
                end
            end
        end
    end

    # Search for IC-violating Uniformly wired boxes:
    print2log("Searching for IC-violating wirings...")
    results["IC_violating_wirings"] = Any[]

    if config.box_search_space == :mid_mid_point

        middle_points_IC_Q = results["quantum_primary_scores"] + (results["unwired_IC_primary_scores"] - results["quantum_primary_scores"]) ./ 2 

        fixed_nl_point = (results["unwired_IC_secondary_scores"][div(end,2)], middle_points_IC_Q[div(end,2)])
        fixed_α, fixed_β, fixed_γ = Compute_Coeff(P1, P2, P3, fixed_nl_point[1], fixed_nl_point[2], secondary_score)
        fixed_nl_box = fixed_α*P1 + fixed_β*P2 + fixed_γ*P3

        #dist_metric = prepare_NPA_BoxDistance(initial_box)

        if config.mode == :uniform
            #Here we are also interested if we can find multiple, different violating wirings = append!
            append!(results["IC_violating_wirings"], uniform_extremal_wiring_search(fixed_nl_box, config.max_wiring_order, is_NOT_in_IC))
        elseif config.mode == :greedy_lifting
            #Here we only care about whether any violating wiring exists = push!
            push!(results["IC_violating_wirings"], greedy_lifting_extremal_wiring_search(fixed_nl_box, config.max_wiring_order, is_NOT_in_IC, sdp_conditions.nearest_pyNPA_point ))
        else
            error("Unknown mode")
        end

    elseif config.box_search_space == :full_IC_Q_gap
        safety_margin = 2 #units of search_precision

        #Iterate over all points in the IC-Q gap with a certain resolution
        for (c_x_idx, c_x_val) in enumerate(results["quantum_secondary_scores"])
            
            if (results["unwired_IC_primary_scores"][c_x_idx] - results["quantum_primary_scores"][c_x_idx]) <= (2*safety_margin+1)*config.search_precision #Minimal gap width of 2 x safety margin + more than one unit of search precision 
                continue #IC-Q gap needs to be wide enough to avoid false positives by numerical errors
            end

            n_results_before_this_x = length(results["IC_violating_wirings"])
                    
            for c_y_val in range(start=(results["quantum_primary_scores"][c_x_idx] + safety_margin*config.search_precision), step=config.search_precision, stop=(results["unwired_IC_primary_scores"][c_x_idx] - safety_margin*config.search_precision))
                print2log("Searching for a Uniform wiring at x=$(round(c_x_val, digits=3)) and y=$(round(c_y_val, digits=3))...")

                c_fixed_α, c_fixed_β, c_fixed_γ = Compute_Coeff(P1, P2, P3, c_x_val, c_y_val, secondary_score)
                c_fixed_nl_box = c_fixed_α*P1 + c_fixed_β*P2 + c_fixed_γ*P3
                
                if config.mode == :uniform
                    append!(results["IC_violating_wirings"], uniform_extremal_wiring_search(c_fixed_nl_box, config.max_wiring_order, is_NOT_in_IC))
                elseif config.mode == :greedy_lifting
                    push!(results["IC_violating_wirings"], greedy_lifting_extremal_wiring_search(c_fixed_nl_box, config.max_wiring_order, is_NOT_in_IC, sdp_conditions.nearest_pyNPA_point ))
                else
                    error("Unknown mode")
                end
            end

            n_results_after = length(results["IC_violating_wirings"])
            if n_results_after == n_results_before_this_x
                print2log("*No* IC-violating wirings found for x-value $(round(c_x_val, digits=3)) in the IC-Q gap.")
            end
        end
    else
        error("Unknown box_search_space")
    end

    return results
end

