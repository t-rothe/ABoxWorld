
include(scriptsdir("WiringsSearch", "wirings_search.jl"))

using Parameters

@with_kw struct WiringsRandomSearchConfig
    mode::Symbol; @assert mode in [:uniform, :greedy_lifting]
    approach_from::Symbol; @assert approach_from in [:below, :above]
    search_precision::Float64
    precision::Float64
    max_wiring_order::Int
end

#Adapt savename() for WiringsRandomSearchConfig
DrWatson.default_prefix(c::WiringsRandomSearchConfig) = string(c.mode)*"_WiringsRandomSearch_"*string()*"_MaxOrder_"*string(c.max_wiring_order)
DrWatson.allaccess(::WiringsRandomSearchConfig) = tuple()


function Base.show(io::IO, p::WiringsRandomSearchConfig)
    println(io, "WiringsRandomSearchConfig:")
    for field_key in fieldnames(WiringsRandomSearchConfig)
        println(io, "  ", field_key, ": ", getfield(p, field_key))
    end 
end

function search_at_fixed_point(fixed_nl_box::Array{Float64, 4}, wiring_search_mode::Symbol, max_wiring_order::Int, IC_violation_criterion::Function)

    if wiring_search_mode == :uniform
        #Here we are also interested if we can find multiple, different violating wirings = append!
        return uniform_extremal_wiring_search(fixed_nl_box, max_wiring_order, IC_violation_criterion)
    elseif wiring_search_mode == :greedy_lifting
        #Here we only care about whether any violating wiring exists = push!
        #distance_metric = sdp_conditions.min_distance_to_pyNPA 
        distance_metric = sdp_conditions.randomization_distance_to_NPA
        return greedy_lifting_extremal_wiring_search(fixed_nl_box, max_wiring_order, IC_violation_criterion, distance_metric)
    else
        error("Unknown wiring search mode")
    end
end




function Wirings_Random_Search(config::WiringsRandomSearchConfig; verbose::Bool=false)

    results = Dict(string(key) => getfield(config, key) for key in fieldnames(WiringsRandomSearchConfig))
   
    # Search for IC-violating wired boxes:
    print2log("Searching for IC-violating wirings...")
    results["IC_violating_wirings"] = Any[]

    safety_margin = 1 #units of search precision (so try to keep search_precision it small!)

    max_num_trails = 4
    box_cnt = 0
    for trail in 1:max_num_trails
        init_box_inst, init_box_decomp = Random_NS_Mixture_CHSH(;non_local_bias=true, return_decomp=true)
        init_box = nsboxes.reconstructFullJoint(init_box_inst)

        if config.approach_from == :above #Starting post-IC and moving into IC-Q gap towards Q

            # Setup the box to search wirings from. 
            #--------------------------------
            while sdp_conditions.is_in_NPA(init_box; level=3)
                #Reinitalize; Do not count as a trail
                init_box_inst, init_box_decomp = Random_NS_Mixture_CHSH(;non_local_bias=true, return_decomp=true)
                init_box = nsboxes.reconstructFullJoint(init_box_inst)
            end

            η = 1.0
            while !is_in_IC( (1-η)*init_box + η*MaxMixedBox )
                box_0 = η*init_box + (1-η)*MaxMixedBox #TODO: Could replace maxmixedbox by randomly sampled LD-box
                η -= config.search_precision
            end

            #η_ahead := Look ahead to see if there is a wide enough IC-Q gap  Required to see whether we land outside of IC; Might be if the shear along IC boundary. .\_/.
            η_ahead = η - (2*safety_margin+1)*config.search_precision
            if sdp_conditions.is_in_NPA( η_ahead*init_box + (1-η_ahead)*MaxMixedBox; level=3)
                continue #Line had (almost) no IC-Q gap; Sample a different line
            elseif !is_in_IC( η_ahead*init_box + (1-η_ahead)*MaxMixedBox )
                @warn "Noticed strange movement, non-IC -> IC -> non-IC, only by adding uniform noise; When should this happen?; For safety, we'll just search for something else"
                continue
            end

            η_safe = η
            η_backwards = η + safety_margin*config.search_precision
            if !is_in_IC( η_backwards*init_box + (1-η_backwards)*MaxMixedBox )
                @warn "Too near to IC boundary, go further into IC-Q gap!"
                η_safe -= safety_margin*config.search_precision 
            end

            box_0 = η_safe*init_box + (1-η_safe)*MaxMixedBox
            box_cnt += 1
            
            print2log("Found $box_cnt-th box in IC-Q gap in trail $trail of $max_num_trails")
            @info "The decomposition of the box is:"
            println((init_box_decomp[1].first[1], Dict(pairs(init_box_decomp[1].first[end]))))
            println((init_box_decomp[2].first[1], Dict(pairs(init_box_decomp[2].first[end]))))
            println((init_box_decomp[3].first[1], Dict(pairs(init_box_decomp[3].first[end]))))
            println("with coefficients: $(round.(init_box_decomp[1].second, digits=3)) $(round.(init_box_decomp[2].second, digits=3)) $(round.(init_box_decomp[3].second, digits=3))")
            
            @assert is_in_IC(box_0) #Last safety check; TODO: remove later
            @assert !sdp_conditions.is_asymp_in_NPA(box_0; level=3)
            

            #Success; Found a box that is in a IC-Q gap; Start search ...
            #--------------------------------
            
            print2log("Starting search for IC-violating wirings for this box in the IC-Q gap ...")
            num_result_before = length(results["IC_violating_wirings"])
            append!(results["IC_violating_wirings"], search_at_fixed_point(box_0, config.mode, config.max_wiring_order, is_NOT_in_IC))
            if length(results["IC_violating_wirings"]) == num_result_before
                print2log("*No* IC-violating wirings found for $box_cnt-th box in IC-Q gap in trail $trail of $max_num_trails")
            end

        elseif config.approach_from == :below #Starting in Q and moving into IC-Q gap
            error("Not implemented yet")
        else
            error("Invalid specification for approach_from")
        end 
    end
    return results
end
