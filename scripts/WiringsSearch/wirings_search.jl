using LinearAlgebra
using TensorOperations
using Distances
using Parameters
using Dates

#Print NamedTuples in a vertical way, rather than horizontal, for logging purposes
Base.show(io::IO, nt::NamedTuple) = begin
    print(io, "(")
    isempty(nt) || print(io, "\n")
    for (i, (key, val)) in enumerate(pairs(nt))
        print(io, " ", key, " = ")
        show(io, val)
        i != length(nt) && print(io, ",\n")
    end
    isempty(nt) || print(io, "\n")
    print(io, ")")
end

function print2log(logmessage::String)
    # current time
    time = Dates.format(now(UTC), dateformat"yyyy-mm-dd HH:MM:SS")

    # memory the process is using
    maxrss = "$(round(Sys.maxrss()/1048576, digits=2)) MiB"

    logdata = (;message=logmessage, # some super important progress update
                maxrss) # lastly the amount of memory being used

    @info savename(time, logdata; connector=" | ", equals=" = ", sort=false, digits=2)
end

function Compute_Coeff(P1::Array{Float64,4}, P2::Array{Float64,4}, P3::Array{Float64,4}, alt_G_score_val::Real, CHSH_score_val::Real, alt_Game_score::Function) # P1, P2, P3 are 2x2x2x2 tensors
    A = [alt_Game_score(P1) alt_Game_score(P2) alt_Game_score(P3);
           CHSH_score(P1)     CHSH_score(P2)     CHSH_score(P3);
                1                    1                1           ]
    b = [alt_G_score_val, CHSH_score_val, 1]
    return A \ b # Equiv. to np.linalg.solve(A, b)
end


function is_in_NPA_TLM(alt_G_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, alt_Game_score::Function)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, alt_G_score_val, CHSH_score_val, alt_Game_score)
    Box = α*Box1 + β*Box2 + γ*Box3
    # Box is a 2x2x2x2 tensor
    #return conditions.check(NPA_TLM_Criterion(), Box)

    E_xy = ProbBellCorrelator(Box)
    coeff00 = asin(E_xy[1,1])
    coeff01 = asin(E_xy[1,2])
    coeff10 = asin(E_xy[2,1])
    coeff11 = asin(E_xy[2,2])
    return coeff00 + coeff01 + coeff10 - coeff11 <= pi
end



function is_in_NPA(alt_G_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, alt_Game_score::Function, level::Int)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, alt_G_score_val, CHSH_score_val, alt_Game_score)
    Box = α*Box1 + β*Box2 + γ*Box3
    # Box is a 2x2x2x2 tensor
    return sdp_conditions.is_in_NPA(Box; level=level, verbose=false)    
end



function is_asymp_in_pyNPA(alt_G_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, alt_Game_score::Function, level::Int)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, alt_G_score_val, CHSH_score_val, alt_Game_score)
    Box = α*Box1 + β*Box2 + γ*Box3
    # Box is a 2x2x2x2 tensor
    return sdp_conditions.is_asymp_in_pyNPA(Box; level=level, verbose=false)
end


function is_in_pyNPA(alt_G_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, alt_Game_score::Function, level::Int)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, alt_G_score_val, CHSH_score_val, alt_Game_score)
    Box = α*Box1 + β*Box2 + γ*Box3
    # Box is a 2x2x2x2 tensor
    return sdp_conditions.is_in_pyNPA(Box; level=level, verbose=false)
end

function min_distance_to_pyNPA(alt_G_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, alt_Game_score::Function, level::Int)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, alt_G_score_val, CHSH_score_val, alt_Game_score)
    p_obs = α*Box1 + β*Box2 + γ*Box3
    # Box is a 2x2x2x2 tensor
    return sdp_conditions.min_distance_to_pyNPA(p_obs; level=level, verbose=false)
end

    
function nearest_pyNPA_point(alt_G_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, alt_Game_score::Function, level::Int)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, alt_G_score_val, CHSH_score_val, alt_Game_score)
    p_obs = α*Box1 + β*Box2 + γ*Box3
    # Box is a 2x2x2x2 tensor
    return sdp_conditions.nearest_pyNPA_point(p_obs; level=level, verbose=false)
end

is_NOT_in_IC(P::Array{Float64,4}) = !(conditions.check(Original_IC_Bound(), P, :Q))
#is_NOT_in_IC(P::Array{Float64,4}) = !(conditions.check(Generalized_Original_IC_Bound(), P, :Q))

function is_NOT_in_IC(secondary_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, secondary_score::Function)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, secondary_score_val, CHSH_score_val, secondary_score)
    MixedBox = α*Box1 + β*Box2 + γ*Box3
    
    return is_NOT_in_IC(MixedBox)
    #return conditions.check(Generalized_Original_IC_Bound(), MixedBox, :Q)
end

is_NOT_in_Uffink(P::Array{Float64,4}) = !(conditions.check(Uffink_Bipartite_2222_Inequality(), P, :Q))

function is_NOT_in_Uffink(secondary_score_val::Real,CHSH_score_val::Real, Box1::Array{Float64,4}, Box2::Array{Float64,4}, Box3::Array{Float64,4}, secondary_score::Function)
    α, β, γ = Compute_Coeff(Box1, Box2, Box3, secondary_score_val, CHSH_score_val, secondary_score)
    MixedBox = α*Box1 + β*Box2 + γ*Box3
    
    return is_NOT_in_Uffink(MixedBox)
    #return conditions.check(Uffink_Bipartite_2222_Inequality(), MixedBox, :Q)
end

# ----------------- #
# ----------------- #


function prepare_NPA_BoxDistance(initial_P::Array{Float64, 4}; membership_level::Int=3, distance_level::Int=2)
    #initial_distance = sdp_conditions.min_distance_to_pyNPA(initial_P; level=distance_level)
    initial_distance = sdp_conditions.randomization_distance_to_NPA(initial_P; level=membership_level)
    function NPA_BoxDistance(target_P::Array{Float64, 4})
        if !sdp_conditions.is_in_NPA(target_P; level=membership_level)
            #return sdp_conditions.min_distance_to_pyNPA(target_P; level=distance_level) # - initial_distance
            return sdp_conditions.randomization_distance_to_NPA(target_P; level=membership_level) - initial_distance
        else
            return 0.0
        end
    end
    return NPA_BoxDistance
end



## Uniform search:
function find_uniformly_wired_sufficient_box_in_orbits(P::Array{Float64, 4}, W_vec::Vector{Float64}, max_wiring_order::Int, stopping_condition::Function)    
    """ Note: Here we want to find a sufficient "box", not wiring (which is fixed in this subroutine)
        If there is a notion of a "sufficiently good" box, then we can stop early; Not always neccesary to compute orbits up to max_wiring_order. """
    
    W = reshape(W_vec,:,1)  #Bevause of the nature of the tensorized boxproduct, W must be in batched form
    
    if stopping_condition(P) #0-th order orbit is already good enough
        return P, 0 #best violating box, order
    else
        # Look at all possible ways to multiply P with itself via W:
        Qs = (P, P, P) #Qright, Qcenter, Qleft
    
        for c_order in 1:max_wiring_order
            Qs = (reduc_BoxProduct(W, Qs[1], P), reduc_BoxProduct(W, Qs[2], Qs[2]), reduc_BoxProduct(W, P, Qs[3]))  #Qright, Qcenter, Qleft
        
            sufficient_box_inds = findall(stopping_condition, Qs)
            if !isempty(sufficient_box_inds)
                sufficient_idx = sufficient_box_inds[1] #Just take the first one, i.e. prefer right multiplication = highest CHSH value
                sufficient_box = Qs[sufficient_idx]
                return sufficient_box, c_order
            end
        end

        # If we reach here, then we have not found a sufficient box
        return missing, missing

    end
end




# ----------------- #
# ----------------- #




function uniform_extremal_wiring_search(initial_box::Array{Float64,4}, max_wiring_order::Int, IC_violation_criterion::Function, wires_generator::Function=extremal_wires_generator)
   
    IC_violating_wirings = Any[]
    
    for (w_i, (c_extremal_wires, c_extremal_wiring_types, c_extremal_wiring_params)) in enumerate(wires_generator())
        #c_extremal_wires = extremal_wires_dict[(c_extremal_wiring_types[1], c_extremal_wiring_params_pair[1], c_extremal_wiring_types[2], c_extremal_wiring_params_pair[2])]
        println("Currently wiring, no. : $w_i")
        IC_viol_wired_box, viol_wiring_order = find_uniformly_wired_sufficient_box_in_orbits(initial_box, c_extremal_wires, max_wiring_order, IC_violation_criterion)
        
        if !ismissing(IC_viol_wired_box)
            print2log("Found IC-violating uniformly wired box at order $viol_wiring_order")
            push!(IC_violating_wirings, (initial_box=initial_box, wired_box=IC_viol_wired_box, wiring_types=c_extremal_wiring_types, wiring_params=c_extremal_wiring_params, wiring_order=viol_wiring_order))
        end
    end
   return IC_violating_wirings
end




function max_in_CHSH_family(Q::Array{Float64,4})
    scores_vals = []
    for s in Iterators.product(([-1, 1] for _ in 1:4)...)
        if prod(s) == -1 #Only allow valid CHSH functionals (= odd number of negative coefficient)
            CHSH_functional = games.CHSH_score_generator(s...; batched=false)	
            #println("CHSH($(s[1]), $(s[2]), $(s[3]), $(s[4])) = ", CHSH_functional(Q))
            push!(scores_vals, CHSH_functional(Q))  
        end
    end
    return maximum(scores_vals)
end

function select_best_wiring(current_best::Union{NamedTuple, Missing}, candidate::NamedTuple)
    """Accompying utility function for greedy_lifting_extremal_wiring_search
    Tuples contain (Wiring NamedTuple, CHSH_val, scoring_function_val)
    Wirings NamedTuple features types, params, associativity
    """
    if ismissing(current_best) #If no current best
        return candidate
    elseif !ismissing(candidate.score) && !ismissing(current_best.score)
        if candidate.score > current_best.score
            return candidate
        else
            return current_best
        end
    elseif ismissing(candidate.score) && !ismissing(current_best.score)
        return current_best
    elseif ismissing(current_best.score) && !ismissing(candidate.score)
        return candidate
    else
        if candidate.chsh > current_best.chsh
            return candidate
        else
            return current_best
        end
    end
end


function greedy_lifting_extremal_wiring_search(initial_box::Array{Float64,4}, max_wiring_order::Int, stopping_condition::Function, scoring_functional::Function)
    """...
    Scoring functional (/distance metric) w.r.t. quantum set: defines at each (non-uniform) wiring order which wiring is best for subsequent wiring orders = greedy search.
    If  by CHSH value.
    """

    if stopping_condition(initial_box) #0-th order orbit is already good enough
        print2log("Found *unwired* IC-violating box. This is unexpected; Is the box search space correctly specified?")
        return (initial_box=initial_box, wired_box=initial_box, wiring_series=[missing, ], wiring_order=0)   
    end
    # Look at all possible ways to multiply P with itself via W:
    Qs = (initial_box, initial_box, initial_box) #Qright, Qcenter, Qleft
    
    associativities = [:R, :C, :L]
    best_wirings = Any[missing,] #Stores at each order (NamedTuple(types=types_tuple, params=params_tuple, associativity ∈ (:R, :C, :L)), CHSH_val, scoring_function_val)
    for c_order in 1:max_wiring_order
        push!(best_wirings, missing)
        for c_extremal_wiring_types in Iterators.product(keys(extremal_wiring_params), keys(extremal_wiring_params))
            for c_extremal_wiring_params_pair in Iterators.product(Iterators.product((0:1 for _ in 1:length(extremal_wiring_params[c_extremal_wiring_types[1]]))...), Iterators.product((0:1 for _ in 1:length(extremal_wiring_params[c_extremal_wiring_types[2]]))...))
                
                c_W = reshape(extremal_wires_dict[(c_extremal_wiring_types[1], c_extremal_wiring_params_pair[1], c_extremal_wiring_types[2], c_extremal_wiring_params_pair[2])], :,1)  #Put in (32,1) matrix bec. of the nature of the tensorized boxproduct, W must be in batched form
                Q_candidates = (reduc_BoxProduct(c_W, Qs[1], initial_box), reduc_BoxProduct(c_W, Qs[2], Qs[2]), reduc_BoxProduct(c_W, initial_box, Qs[3]))  #Qright, Qcenter, Qleft
                
                sufficient_box_inds = findall(stopping_condition, Q_candidates)
                if !isempty(sufficient_box_inds)
                    sufficient_idx = sufficient_box_inds[1] #Just take the first one, i.e. prefer right multiplication = highest CHSH value
                    sufficient_box = Q_candidates[sufficient_idx]
                    print2log("Found IC-violating wired box at order $c_order")
                    found_wiring_series = [[best_wirings[i][1] for i in 1:(c_order-1)]; [(types=c_extremal_wiring_types, params=c_extremal_wiring_params_pair, associatity=associativities[sufficient_idx]), ]]
                    return [(initial_box=initial_box, wired_box=sufficient_box, wiring_series=found_wiring_series, wiring_order=c_order), ]
                end

                # We have not found a sufficient box for this wiring and order, so update best wiring
                chsh_scores = [CHSH_score(Q_candidate) for Q_candidate in Q_candidates]
                #@show chsh_scores
                #dist_scores = [sdp_conditions.randomization_distance_to_NPA_boundary(Q_candidate; level=3) for Q_candidate in Q_candidates]
                #@show dist_scores
                IC_scores = [IC_Bound_LHS(Q_candidate) for Q_candidate in Q_candidates]
                chsh_scores = IC_scores
                #@show IC_scores
                #chsh_scores = [max_in_CHSH_family(Q_candidate) for Q_candidate in Q_candidates]
                
                non_quantum_Q_inds = findall(!sdp_conditions.is_in_NPA, Q_candidates)
                #@show c_extremal_wiring_types,chsh_scores, non_quantum_Q_inds
                
                #Determine best associativity of wiring
                if isempty(non_quantum_Q_inds) #all quantum -> CHSH determines best
                    (max_chsh_val, max_chsh_box_idx) = findmax(chsh_scores)   
                    wiring_candidate = (wiring=(types=c_extremal_wiring_types, params=c_extremal_wiring_params_pair, associativity=associativities[max_chsh_box_idx]), chsh=max_chsh_val, score=missing)
                    best_wirings[c_order] = select_best_wiring(best_wirings[c_order], wiring_candidate)
                
                elseif length(non_quantum_Q_inds) == 1
                    wiring_candidate = (wiring=(types=c_extremal_wiring_types, params=c_extremal_wiring_params_pair, associativity=associativities[non_quantum_Q_inds[1]]), chsh=chsh_scores[non_quantum_Q_inds[1]], score=missing)
                    best_wirings[c_order] = select_best_wiring(best_wirings[c_order], wiring_candidate)

                else #-> scoring function decides if more than one non-quantum Q candidate
                    print2log("Since we got post-quantum CHSH values $(chsh_scores), we need to use scoring function to decide. This will probably take a while ...")
                    non_quantum_Q_candidates = [Q_candidates[i] for i in non_quantum_Q_inds]
                    (max_score_val, max_score_idx) = findmax(scoring_functional, non_quantum_Q_candidates)   
                    max_score_original_idx = non_quantum_Q_inds[max_score_idx]
                    wiring_candidate = (wiring=(types=c_extremal_wiring_types, params=c_extremal_wiring_params_pair, associativity=associativities[max_score_original_idx]), chsh=chsh_scores[max_score_original_idx], score=max_score_val)
                    best_wirings[c_order] = select_best_wiring(best_wirings[c_order], wiring_candidate)
                end
            end
        end
        print2log("Finished wiring order $c_order ; Nothing found yet. Best series of wirings so far:")
        for i in 1:c_order
            @info best_wirings[i]
        end

        #Prepare Qs for next order
        best_types, best_params = best_wirings[c_order][1].types, best_wirings[c_order][1].params
        best_W = reshape(extremal_wires_dict[(best_types[1], best_params[1], best_types[2], best_params[2])], :,1) 
        newQs = (reduc_BoxProduct(best_W, Qs[1], initial_box), reduc_BoxProduct(best_W, Qs[2], Qs[2]), reduc_BoxProduct(best_W, initial_box, Qs[3]))  #Qright, Qcenter, Qleft
        Qs = newQs # overwrite
    end
    
    #Found nothing, stopping after reaching max_wiring_order
    return [missing, ]
 end

# ----------------- #
# ----------------- #
 
