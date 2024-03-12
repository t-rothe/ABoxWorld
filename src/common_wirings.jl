!isdefined(Main, :wirings) ? (include("wirings.jl"); using .wirings) : nothing
using IterTools
using LinearAlgebra


function random_wire_matrix(n::Int)  # n is the number of columns
    return rand(32, n)
end

function random_wiring() 
    return wirings.Wiring((2,2,2,2), rand(32))
end

function random_deterministic_wire_matrix(n::Int)  # n is the number of columns
    return rand(0:1, 32, n)
end

function random_deterministic_wiring() 
    return wirings.Wiring((2,2,2,2), rand(0:1, 32))
end


function extremal_wires(wire_type_A::Symbol, wire_params_A::Dict{Symbol,<:Int}, wire_type_B::Symbol, wire_params_B::Dict{Symbol,<:Int})
    @assert (wire_type_A in [:D, :O, :X, :A, :S]) "Invalid extremal wire type for A"
    @assert (wire_type_B in [:D, :O, :X, :A, :S]) "Invalid extremal wire type for B"
    
    @assert all([x in [0,1] for x in values(wire_params_A)]) "Invalid extremal wire parameters for A"
    @assert all([x in [0,1] for x in values(wire_params_B)]) "Invalid extremal wire parameters for B"

    wire_buffer = []
    for (wire_type, params) in zip((wire_type_A, wire_type_B), (wire_params_A, wire_params_B))
        f1, f2, f3 = zeros(2,2), zeros(2,2), zeros(2,2,2)
        if wire_type == :D
            f3 .= params[:α]
        elseif wire_type == :O
            f1 .= params[:α]
            f2 .= params[:α]
            if params[:β] == 0
                for a1 in 0:1
                    f3[:,a1+1,:] .= (a1 ⊻ params[:γ])
                end
            else #β == 1
                for a2 in 0:1
                    f3[:,:,a2+1] .= (a2 ⊻ params[:γ])
                end
            end
        elseif wire_type == :X
            f1 .= params[:α]
            f2 .= params[:β]
            for a2 in 0:1
                for a1 in 0:1
                    f3[:,a1+1,a2+1] .= (a1 ⊻ a2 ⊻ params[:γ])
                end
            end
        elseif wire_type == :A
            f1 .= params[:α]
            f2 .= params[:β]
            for a2 in 0:1
                for a1 in 0:1
                    f3[:,a1+1,a2+1] .= ((a1 ⊻ params[:γ])*(a2 ⊻ params[:δ]) ⊻ params[:ε])
                end
            end
        elseif wire_type == :S
            if params[:α] == 0
                f1 .= params[:β]
                for a1 in 0:1
                    f2[:,a1+1] .= (a1 ⊻ params[:γ])
                end
                for a2 in 0:1
                    for a1 in 0:1
                        f3[:,a1+1,a2+1] .= ((params[:δ]*a1) ⊻ a2 ⊻ params[:ε])
                    end
                end
            else #α == 1
                for a2 in 0:1
                    f2[:,a2+1] .= (a2 ⊻ params[:γ])
                end
                f2 .= params[:β]
                for a2 in 0:1
                    for a1 in 0:1
                        f3[:,a1+1,a2+1] .= (a1 ⊻ (params[:δ]*a2) ⊻ params[:ε])
                    end
                end
            end
        else 
            error("Unknown wire type $(wire_type)")
        end
        push!(wire_buffer, (f1, f2, f3))
    end
    return wirings.functions_to_wires(wire_buffer[1][1], wire_buffer[2][1], wire_buffer[1][2], wire_buffer[2][2], wire_buffer[1][3], wire_buffer[2][3])
end

function collect_all_extremal_wires()
    """Compute all 6,724 = 82^2 extremal wires, each a vector of dimension 32."""
    
    wire_types_and_params = Dict(:D => [:α,], 
                                :O => [:α, :β, :γ], 
                                :X => [:α, :β, :γ],
                                :A => [:α, :β, :γ, :δ, :ε],
                                :S => [:α, :β, :γ, :δ, :ε],
                                )

    wire_collection = Dict() # 82 wires, each a vector of dimension 32
    for (wire_type_B, wire_param_names_B) in wire_types_and_params
        for (wire_type_A, wire_param_names_A) in wire_types_and_params
            temp_wire_buffer = []
            for wire_param_vals_B in IterTools.product((0:1 for _ in 1:length(wire_param_names_B))...)
                wire_params_B = Dict(zip(wire_param_names_B, wire_param_vals_B))
                for wire_param_vals_A in IterTools.product((0:1 for _ in 1:length(wire_param_names_A))...)
                    wire_params_A = Dict(zip(wire_param_names_A, wire_param_vals_A))
                    push!(temp_wire_buffer, extremal_wires(wire_type_A, wire_params_A, wire_type_B, wire_params_B))
                end
            end
            wire_collection[(wire_type_A, wire_type_B)] = transpose(stack(temp_wire_buffer, dims=1))
        end
    end

    return wire_collection
end