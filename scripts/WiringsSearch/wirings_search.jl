using LinearAlgebra
using TensorOperations
using Distances

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


# ----------------- #
# ----------------- #


function prepare_NPA_BoxDistance(initial_P::Array{Float64, 4}; membership_level::Int=3, distance_level::Int=2)
    #initial_distance = sdp_conditions.min_distance_to_pyNPA(initial_P; level=distance_level)
    function NPA_BoxDistance(target_P::Array{Float64, 4})
        if !sdp_conditions.is_in_NPA(target_P; level=membership_level)
            return sdp_conditions.min_distance_to_pyNPA(target_P; level=distance_level) # - initial_distance
        else
            return 0.0
        end
    end
    return NPA_BoxDistance
end



function greedy_sufficient_box_in_orbits(P::Array{Float64, 4}, W_vec::Vector{Float64}, max_wiring_order::Int, stopping_condition::Function)    
    """ If there is a notion of a "sufficiently good" box, then we can stop early; Not always neccesary to compute orbits up to max_wiring_order. """
    
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

#Extremal wire types & parameters (binary):
extremal_wiring_params = Dict(:D => [:α,], 
:O => [:α, :β, :γ], 
:X => [:α, :β, :γ],
:A => [:α, :β, :γ, :δ, :ε],
:S => [:α, :β, :γ, :δ, :ε],
)

#Precompute the extremal wirings in the Dict format of extremal_wiring_params
extremal_wires_dict = Dict()
for (wiretype_A, wiretype_B) in Iterators.product(keys(extremal_wiring_params), keys(extremal_wiring_params))
   for c_wire_params_B in Iterators.product((0:1 for _ in 1:length(extremal_wiring_params[wiretype_B]))...)
       for c_wire_params_A in Iterators.product((0:1 for _ in 1:length(extremal_wiring_params[wiretype_A]))...)
           extremal_wires_dict[(wiretype_A, c_wire_params_A, wiretype_B, c_wire_params_B)] = extremal_wires(wiretype_A, Dict(zip(extremal_wiring_params[wiretype_A], c_wire_params_A)), wiretype_B, Dict(zip(extremal_wiring_params[wiretype_B], c_wire_params_B)))
       end
   end
end


function uniform_extremal_wiring_search(initial_box::Array{Float64,4}, max_wiring_order::Int, IC_violation_criterion::Function)
   
   IC_violating_wirings = []

   for c_extremal_wiring_types in Iterators.product(keys(extremal_wiring_params), keys(extremal_wiring_params))
       for c_extremal_wiring_params_pair in Iterators.product(Iterators.product((0:1 for _ in 1:length(extremal_wiring_params[c_extremal_wiring_types[1]]))...), Iterators.product((0:1 for _ in 1:length(extremal_wiring_params[c_extremal_wiring_types[2]]))...))
           
           c_extremal_wires = extremal_wires_dict[(c_extremal_wiring_types[1], c_extremal_wiring_params_pair[1], c_extremal_wiring_types[2], c_extremal_wiring_params_pair[2])]
           
           IC_viol_wired_box, viol_wiring_order = greedy_sufficient_box_in_orbits(initial_box, c_extremal_wires, max_wiring_order, IC_violation_criterion)
           if !ismissing(IC_viol_wired_box)
               @info "Found IC-violating uniformly wired box at order $viol_wiring_order"
               push!(IC_violating_wirings, (initial_box=initial_box, wired_box=IC_viol_wired_box, wiring_types=c_extremal_wiring_types, wiring_params=c_extremal_wiring_params_pair, wiring_order=viol_wiring_order))
           end
       end
   end
   return IC_violating_wirings
end

