
include(scriptsdir("WiringsSearch", "wirings_search.jl"))


@with_kw struct WiringsSliceSearchConfig
    mode::Symbol; @assert mode in [:uniform, :single, :greedy_lifting, :collect_plot_data]
    box_search_space::Symbol; @assert box_search_space in [:mid_mid_point, :point_near_IC_boundary,:full_IC_Q_gap, :below_IC_boundary]
    Box1::Pair{String, Array{Float64,4}}
    Box2::Pair{String, Array{Float64,4}}
    Box3::Pair{String, Array{Float64,4}}
    primary_score::Function = CHSH_score
    secondary_score::Function 
    IC_violation_criterion::Function = is_NOT_in_IC
    boundary_precision::Float64
    search_precision::Float64; @assert search_precision >= boundary_precision #Search can't be more precise than the computed boundaries
    precision::Float64
    wires_generator::Union{Function, Channel} = extremal_wires_generator
    max_wiring_order::Int
end

#Adapt savename() for WiringsSliceSearchConfig:
DrWatson.default_prefix(c::WiringsSliceSearchConfig) = string(c.mode)*"_WiringsSliceSearch_"*string(c.box_search_space)*"_"*c.Box1.first*"_"*c.Box2.first*"_"*c.Box3.first*"_MaxOrder_"*string(c.max_wiring_order)
DrWatson.allaccess(::WiringsSliceSearchConfig) = tuple()


function Base.show(io::IO, p::WiringsSliceSearchConfig)
    println(io, "WiringsSliceSearchConfig:")
    for field_key in fieldnames(WiringsSliceSearchConfig)
        println(io, "  ", field_key, ": ", getfield(p, field_key))
    end 
end



function search_at_fixed_point(fixed_primary_score_val::Float64, fixed_secondary_score_val::Float64, Box1::Array{Float64, 4}, Box2::Array{Float64, 4}, Box3::Array{Float64, 4}, secondary_score::Function, wiring_search_mode::Symbol, max_wiring_order::Int, wires_generator::Union{Function, Channel}, IC_violation_criterion::Function)
    fixed_α, fixed_β, fixed_γ = Compute_Coeff(Box1, Box2, Box3, fixed_primary_score_val, fixed_secondary_score_val, secondary_score)
    fixed_nl_box = fixed_α*Box1 + fixed_β*Box2 + fixed_γ*Box3

    if wiring_search_mode == :uniform
        #Here we are also interested if we can find multiple, different violating wirings = append!
        return uniform_extremal_wiring_search(fixed_nl_box, max_wiring_order, IC_violation_criterion, wires_generator)
    elseif wiring_search_mode == :greedy_lifting
        #Here we only care about whether any violating wiring exists = push!
        #distance_metric = sdp_conditions.min_distance_to_pyNPA 
        distance_metric = sdp_conditions.randomization_distance_to_NPA
        return greedy_lifting_extremal_wiring_search(fixed_nl_box, max_wiring_order, IC_violation_criterion, distance_metric)
    elseif wiring_search_mode == :single
        
        IC_viol_wired_box, viol_wiring_order = find_uniformly_wired_sufficient_box_in_orbits(fixed_nl_box, wires_generator(), max_wiring_order,  IC_violation_criterion)
        if !ismissing(IC_viol_wired_box)
            print2log("Found IC-violating uniformly wired box at order $viol_wiring_order")
            return Any[(initial_box=fixed_nl_box, wired_box=IC_viol_wired_box, wiring_types=missing, wiring_params=missing, wiring_order=viol_wiring_order), ]
        end
        return Any[]
    else
        error("Unknown wiring search mode")
    end
end



function Wirings_Slice_Search(config::WiringsSliceSearchConfig; verbose::Bool=false)

    results = Dict(string(key) => getfield(config, key) for key in fieldnames(WiringsSliceSearchConfig))

    is_NOT_in_IC = config.IC_violation_criterion

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
                    #push!(results["unwired_IC_primary_scores"], missing) #Add the NS boundary point to the list
                    push!(results["unwired_IC_primary_scores"], algebraic_max_CHSH_score - c_secondary_score_val + min_secondary_score) #Add the NS boundary point to the list
                    c_CHSH_score_val = (algebraic_max_CHSH_score - max(x2, x3))*(c_secondary_score_val-min(x2, x3))/(max(x2, x3)-min(x2, x3)) + init_CHSH_guess*(c_secondary_score_val-max(x2, x3))/(min(x2,x3)-max(x2, x3)) # Lagrange interpolation; Same as in comment above
                    break
                end
            end
        end
    end


    # Search for IC-violating Uniformly wired boxes:
    print2log("Searching for IC-violating wirings...")
    
    results["IC_violating_wirings"] = Any[]
    if config.mode == :collect_plot_data
        results["assessed_points"] = Any[]
    end

    if config.box_search_space == :mid_mid_point

        middle_points_IC_Q = results["quantum_primary_scores"] + (results["unwired_IC_primary_scores"] - results["quantum_primary_scores"]) ./ 2 

        fixed_nl_point = (results["unwired_IC_secondary_scores"][div(end,2)], middle_points_IC_Q[div(end,2)])
        
        if config.mode != :collect_plot_data
            append!(results["IC_violating_wirings"], search_at_fixed_point(fixed_nl_point[1], fixed_nl_point[2], P1, P2, P3, secondary_score, config.mode, config.max_wiring_order, config.wires_generator, is_NOT_in_IC))
        else
            push!(results["assessed_points"], fixed_nl_point)
        end
    elseif config.box_search_space == :point_near_IC_boundary
        secondary_axis_idx = Int(ceil(3/4*length(results["unwired_IC_secondary_scores"])))
        fixed_nl_point = (results["unwired_IC_secondary_scores"][secondary_axis_idx], results["unwired_IC_primary_scores"][secondary_axis_idx] - 2*config.boundary_precision) 

        
        if config.mode != :collect_plot_data
            append!(results["IC_violating_wirings"], search_at_fixed_point(fixed_nl_point[1], fixed_nl_point[2], P1, P2, P3, secondary_score, config.mode, config.max_wiring_order, config.wires_generator, is_NOT_in_IC))
        else
            push!(results["assessed_points"], fixed_nl_point)
        end

    elseif config.box_search_space == :full_IC_Q_gap
        safety_margin = 2 #units of boundary_precision

        config.mode == :collect_plot_data && (results["assessed_points"] = [])
        #Iterate over all points in the IC-Q gap with a certain resolution
        for (c_x_idx, c_x_val) in enumerate(results["quantum_secondary_scores"])
            
            if mod(c_x_idx, Int(floor(config.search_precision/config.boundary_precision))) != 0 || (results["unwired_IC_primary_scores"][c_x_idx] - results["quantum_primary_scores"][c_x_idx]) <= (2*safety_margin+1)*config.boundary_precision #Minimal gap width of 2 x safety margin + more than one unit of search precision 
                continue #IC-Q gap needs to be wide enough to avoid false positives by numerical errors
            end

            n_results_before_this_x = length(results["IC_violating_wirings"])
                    
            for c_y_val in range(start=(results["quantum_primary_scores"][c_x_idx] + safety_margin*config.boundary_precision), step=config.search_precision, stop=(results["unwired_IC_primary_scores"][c_x_idx] - safety_margin*config.boundary_precision))
                print2log("Searching for a wiring at x=$(round(c_x_val, digits=3)) and y=$(round(c_y_val, digits=3))...")

                if config.mode != :collect_plot_data
                    append!(results["IC_violating_wirings"], search_at_fixed_point(c_x_val, c_y_val, P1, P2, P3, secondary_score, config.mode, config.max_wiring_order, config.wires_generator, is_NOT_in_IC))
                else
                    push!(results["assessed_points"], (c_x_val, c_y_val))
                end
            end

            n_results_after = length(results["IC_violating_wirings"])
            if n_results_after == n_results_before_this_x
                print2log("*No* IC-violating wirings found for x-value $(round(c_x_val, digits=3)) in the IC-Q gap.")
            end
        end
    
    elseif config.box_search_space == :below_IC_boundary
        safety_margin = 1 #units of boundary_precision
        search_strip_width = 1 #units of search_precision

        for c_boundary_dist in 1:search_strip_width
            
            n_results_before_this_strip_row = length(results["IC_violating_wirings"])

            for (c_x_idx, c_x_val) in Iterators.reverse(enumerate(results["quantum_secondary_scores"]))
                if mod(c_x_idx, Int(floor(config.search_precision/config.boundary_precision))) != 0 || (results["unwired_IC_primary_scores"][c_x_idx] - results["quantum_primary_scores"][c_x_idx]) <= safety_margin*config.boundary_precision + c_boundary_dist*config.search_precision  #Minimal gap width of 2 x safety margin + more than one unit of search precision 
                    continue #IC-Q gap needs to be wide enough to avoid false positives by numerical errors
                end

                c_y_val = results["unwired_IC_primary_scores"][c_x_idx] - safety_margin*config.boundary_precision - c_boundary_dist*config.search_precision
                print2log("Searching for a wiring at x=$(round(c_x_val, digits=3)) and y=$(round(c_y_val, digits=3))...")

                
                if config.mode != :collect_plot_data
                    append!(results["IC_violating_wirings"], search_at_fixed_point(c_x_val, c_y_val, P1, P2, P3, secondary_score, config.mode, config.max_wiring_order, config.wires_generator, is_NOT_in_IC))
                else
                    push!(results["assessed_points"], (c_x_val, c_y_val))
                end
            end

            n_results_after_this_strip_row = length(results["IC_violating_wirings"])
            if n_results_after_this_strip_row == n_results_before_this_strip_row
                print2log("*No* IC-violating wirings found for row $(c_boundary_dist) below the IC boundary.")
            end
        end
    else
        error("Unknown box_search_space")
    end

    return results
end

