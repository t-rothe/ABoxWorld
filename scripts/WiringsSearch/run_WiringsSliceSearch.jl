
using DrWatson
@quickactivate "ABoxWorld"
using Pkg; Pkg.instantiate()

include(srcdir("ABoxWorld.jl"));

@info "Project-name = $(projectname()), Julia Depot @ $DEPOT_PATH"

using LinearAlgebra
using TensorOperations
using Parameters


# Select and prepare some utitilies and scoring functions
# -------------------------------------------------------

convert_matrixbox_to_nsjoint = wirings.convert_matrixbox_to_nsjoint
convert_nsjoint_to_matrixbox = wirings.convert_nsjoint_to_matrixbox


# Boxes as 2x2x2x2 tensors
MaxMixedBox = nsboxes.reconstructFullJoint(UniformRandomBox((2, 2, 2, 2)))
PR(μ, ν, σ) = nsboxes.reconstructFullJoint(PRBoxesCHSH(;μ=μ, ν=ν, σ=σ))
CanonicalPR = PR(0, 0, 0)
PL(α, γ, β, λ) = nsboxes.reconstructFullJoint(LocalDeterministicBoxesCHSH(;α=α, γ=γ, β=β, λ=λ))
SR = (PL(0,0,0,0) .+ PL(0,1,0,1)) ./ 2
#SR = matrix_to_tensor(non_local_boxes.utils.SR)
#PRprime = matrix_to_tensor(non_local_boxes.utils.PRprime)
#P0 = matrix_to_tensor(non_local_boxes.utils.P_0)
#P1 = matrix_to_tensor(non_local_boxes.utils.P_1)


CHSH_score = games.canonical_CHSH_score
#CHSHprime_score = games.CHSH_score_generator(-1,1,1,1; batched=false)
CHSHprime_score = games.CHSH_score_generator(1,-1,1,1; batched=false)


BoxProduct = wirings.tensorized_boxproduct
reduc_BoxProduct = wirings.reduc_tensorized_boxproduct
BoxProduct_matrices(w::Matrix{<:Real}, matrixbox1::Matrix{Float64}, matrixbox2::Matrix{Float64}) = convert_nsjoint_to_matrixbox(reduc_BoxProduct(w, matrixbox1, matrixbox2))


IC_Bound = Original_IC_Bound()
IC_Bound_LHS(P::Array{Float64, 4}) = conditions.evaluate(IC_Bound, P)
IC_Bound_LHS(W::Matrix{<:Real}, P_mat::Matrix{Float64}, Q_mat::Matrix{Float64}) = games.Original_IC_Bound_score(W, P_mat, Q_mat)
IC_Bound_LHS(W::Vector{<:Real}, P_mat::Matrix{Float64}, Q_mat::Matrix{Float64}) = games.Original_IC_Bound_score(W, P_mat, Q_mat)

IC_MutInfo = games.MutInfo_IC_vanDam_score


#-------------------------------------------------------
include(scriptsdir("WiringsSearch", "WiringsSliceSearch.jl"))

#-------------------------------------------------------

function main(c_config)
    print(c_config)

    #data_output = Wirings_Slice_Search(c_config; verbose=true)

    data_output, data_filename = produce_or_load(Wirings_Slice_Search, 
                                                c_config,
                                                mkpath(datadir("WiringsSliceSearch"));
                                                verbose=true,
                                                )

    print(data_output)
end
 
if abspath(PROGRAM_FILE) == @__FILE__

    """
    c_config = WiringsSliceSearchConfig(mode=:uniform, 
                            box_search_space=:full_IC_Q_gap, 
                            Box1=("PR"=>CanonicalPR),
                            Box2=("I"=>MaxMixedBox),
                            Box3=("PL(0,0,0,0)"=>PL(0,0,0,0)),
                            primary_score=CHSH_score,
                            secondary_score=CHSHprime_score,
                            boundary_precision=4e-3,
                            search_precision=8e-3,
                            precision=1.4e-2,
                            max_wiring_order=4,
                            )


    c_config = WiringsSliceSearchConfig(mode=:greedy_lifting, #:uniform, #
                                    box_search_space=:mid_mid_point, #:full_IC_Q_gap, #,# , #
                                    Box1=("PR"=>CanonicalPR),
                                    Box2=("I"=>MaxMixedBox),
                                    Box3=("PL(0,0,0,0)"=>PL(0,0,0,0)),
                                    primary_score=CHSH_score,
                                    secondary_score=CHSHprime_score,
                                    boundary_precision=4e-3,
                                    search_precision=4e-3,
                                    precision=1.4e-2,
                                    max_wiring_order=8,
                                    )

    #Command-line equivalent to the above config:
    julia run_WiringsSliceSearch.jl mode=greedy_lifting box_search_space=mid_mid_point boundary_precision=4e-3 search_precision=4e-3 max_wiring_order=10
    and the other fixed point:
    julia run_WiringsSliceSearch.jl mode=greedy_lifting box_search_space=point_near_IC_boundary boundary_precision=4e-3 search_precision=4e-3 max_wiring_order=8
    
    c_config = WiringsSliceSearchConfig(mode=:uniform, #
                                box_search_space=:below_IC_boundary ,# :full_IC_Q_gap, #,# , #
                                Box1=("PR"=>CanonicalPR),
                                Box2=("I"=>MaxMixedBox),
                                Box3=("PL(0,0,0,0)"=>PL(0,0,0,0)),
                                primary_score=CHSH_score,
                                secondary_score=CHSHprime_score,
                                boundary_precision=4e-3,
                                search_precision=1e-2,
                                precision=1.4e-2,
                                max_wiring_order=8,
                                )



                    c_config = WiringsSliceSearchConfig(mode=:uniform,
                                box_search_space=:below_IC_boundary,
                                Box1=("PR"=>CanonicalPR),
                                Box2=("I"=>MaxMixedBox),
                                Box3=("PL(0,1,0,1)"=>PL(0,1,0,1)),
                                primary_score=CHSH_score,
                                secondary_score=CHSHprime_score,
                                IC_violation_criterion = is_NOT_in_IC, 
                                boundary_precision=4e-3,
                                search_precision=8e-3,
                                precision=1.4e-2,
                                wires_generator=Allcock2009_wires_generator,
                                #wires_generator=canonical_extremal_wires_generator,
                                max_wiring_order=10,
                                )
    Command-line equivalent:
    julia run_WiringsSliceSearch.jl mode=uniform box_search_space=below_IC_boundary boundary_precision=4e-3 search_precision=8e-3 max_wiring_order=10
    


    """
    # Initialize a dictionary to hold the named arguments
    named_args = Dict{String, String}()

    # Iterate through each argument passed to the script
    for arg in ARGS
        # Split the argument on the '=' character
        pair = split(arg, "=", limit=2)
        if length(pair) == 2
            # Add the key-value pair to the dictionary
            named_args[pair[1]] = pair[2]
        else
            println("Ignoring malformed argument: $arg")
        end
    end

    c_config = WiringsSliceSearchConfig(mode=Symbol(get(named_args, "mode", missing)), 
                                box_search_space=Symbol(get(named_args, "box_search_space", missing)),
                                Box1=("PR"=>CanonicalPR),
                                Box2=("I"=>MaxMixedBox),
                                Box3=("PL(0,0,0,0)"=>PL(0,0,0,0)),
                                #Box3=("PL(0,1,0,1)"=>PL(0,1,0,1)),
                                IC_violation_criterion = is_NOT_in_IC, 
                                primary_score=CHSH_score,
                                secondary_score=CHSHprime_score,
                                boundary_precision=parse(Float64, get(named_args, "boundary_precision", missing)),
                                search_precision=parse(Float64, get(named_args, "boundary_precision", missing)),
                                precision=1.4e-2,
                                #wires_generator=extremal_wires_generator,
                                #wires_generator=distributed_extremal_wires_generator,
                                #wires_generator=canonical_extremal_wires_generator,
                                wires_generator=Allcock2009_wires_generator,
                                max_wiring_order=parse(Int, get(named_args, "max_wiring_order", missing)),
                                )

    #Example command line call:
    #julia run_WiringsSliceSearch.jl mode=uniform box_search_space=full_IC_Q_gap boundary_precision=4e-3 search_precision=8e-3 max_wiring_order=4

    main(c_config)
end
                       

