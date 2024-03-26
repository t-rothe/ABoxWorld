
using Distributed
addprocs(12)
@everywhere using SharedArrays

using ProgressMeter
@everywhere using DrWatson
@quickactivate "ABoxWorld"
using Pkg; Pkg.instantiate()
@everywhere @quickactivate "ABoxWorld"


@everywhere begin
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
end

#-------------------------------------------------------
@everywhere include(scriptsdir("WiringsSearch", "WiringsSliceSearch.jl"))

# And overwriting any routines that need to be distributed:

wire_types_and_params = Dict(:D => [:α,], 
:O => [:α, :β, :γ], 
:X => [:α, :β, :γ],
:A => [:α, :β, :γ, :δ, :ε],
:S => [:α, :β, :γ, :δ, :ε],
)

function uniform_extremal_wiring_search(initial_box::Array{Float64,4}, max_wiring_order::Int, IC_violation_criterion::Function, wires_generator::Union{Function, Channel}=extremal_wires_generator)
    @everywhere initial_box = $initial_box
    @everywhere max_wiring_order = $max_wiring_order
    @everywhere IC_violation_criterion = $IC_violation_criterion
    
    IC_violating_wirings = SharedArray{Any}[] #Any[]
    
    #pmap(enumerate(wires_generator())) do (w_i, (c_extremal_wires, c_extremal_wiring_types, c_extremal_wiring_params))
    @showprogress "Iterating type combinations of B..." for c_wiretype_paramnames_pairs_B in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...)
        c_wire_types_B, c_wire_param_names_B = Tuple(t[1] for t in c_wiretype_paramnames_pairs_B), Tuple(t[2] for t in c_wiretype_paramnames_pairs_B)
        @everywhere c_wire_types_B, c_wire_param_names_B = $c_wire_types_B, $c_wire_param_names_B
        @showprogress "Iterating type combinations of A..." for c_wiretype_paramnames_pairs_A in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...)
            c_wire_types_A, c_wire_param_names_A = Tuple(t[1] for t in c_wiretype_paramnames_pairs_A), Tuple(t[2] for t in c_wiretype_paramnames_pairs_A)
            @everywhere c_wire_types_A, c_wire_param_names_A = $c_wire_types_A, $c_wire_param_names_A

            pmap(Iterators.product((Iterators.product((0:1 for _ in c_wire_param_names_B[i])...) for i in eachindex(c_wire_types_B))...)) do c_wire_param_vals_B
                #@show c_wire_param_vals_B, c_wire_types_B, c_wire_param_names_B
                c_wire_params_B = Tuple(NamedTuple(zip(c_wire_param_names_B[i], c_wire_param_vals_B[i])) for i in eachindex(c_wire_types_B))
                for c_wire_param_vals_A in Iterators.product((Iterators.product((0:1 for _ in c_wire_param_names_A[i])...) for i in eachindex(c_wire_types_A))...)
                    c_wire_params_A = Tuple(NamedTuple(zip(c_wire_param_names_A[i], c_wire_param_vals_A[i])) for i in eachindex(c_wire_types_A))
                    
                    c_extremal_wires, c_extremal_wiring_types, c_extremal_wiring_params = extremal_wires(c_wire_types_A, c_wire_params_A, c_wire_types_B, c_wire_params_B), (c_wire_types_A, c_wire_types_B), (c_wire_params_A, c_wire_params_B)
                    
                    #println("Currently wiring, no. : $w_i")
                    IC_viol_wired_box, viol_wiring_order = find_uniformly_wired_sufficient_box_in_orbits(initial_box, c_extremal_wires, max_wiring_order, IC_violation_criterion)
                    
                    if !ismissing(IC_viol_wired_box)
                        print2log("Found IC-violating uniformly wired box at order $viol_wiring_order")
                        push!(IC_violating_wirings, (initial_box=initial_box, wired_box=IC_viol_wired_box, wiring_types=c_extremal_wiring_types, wiring_params=c_extremal_wiring_params, wiring_order=viol_wiring_order))
                    end
                end
            end
        end
    end
    return IC_violating_wirings
end

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
                                #Box3=("PL(0,0,0,0)"=>PL(0,0,0,0)),
                                Box3=("PL(0,1,0,1)"=>PL(0,1,0,1)),
                                IC_violation_criterion = is_NOT_in_Uffink, 
                                primary_score=CHSH_score,
                                secondary_score=CHSHprime_score,
                                boundary_precision=parse(Float64, get(named_args, "boundary_precision", missing)),
                                search_precision=parse(Float64, get(named_args, "boundary_precision", missing)),
                                precision=1.4e-2,
                                wires_generator=extremal_wires_generator(),
                                #wires_generator=distributed_extremal_wires_generator,
                                #wires_generator=canonical_extremal_wires_generator,
                                #wires_generator=Allcock2009_wires,
                                max_wiring_order=parse(Int, get(named_args, "max_wiring_order", missing)),
                                )

    #Example command line call:
    #julia run_WiringsSliceSearch.jl mode=uniform box_search_space=full_IC_Q_gap boundary_precision=4e-3 search_precision=8e-3 max_wiring_order=4

    main(c_config)
end
                       

