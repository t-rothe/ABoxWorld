!isdefined(Main, :wirings) ? (include("wirings.jl"); using .wirings) : nothing
using IterTools
using LinearAlgebra
using ProgressMeter
using Serialization
@assert isdefined(Main, :serialize)
#import Pkg; Pkg.add("JLD2")
#using JLD2

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


function unconditional_extremal_wires(wire_type_A::Symbol, wire_params_A::NamedTuple, wire_type_B::Symbol, wire_params_B::NamedTuple)
    """unconditional = does not depend on the input values"""
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

function collect_all_unconditional_extremal_wires()
    """Compute all 6,724 = 82^2 input-independent (=unconditional) extremal wires, each a vector of dimension 32."""
    
    wire_types_and_params = Dict(:D => [:α,], 
                                :O => [:α, :β, :γ], 
                                :X => [:α, :β, :γ],
                                :A => [:α, :β, :γ, :δ, :ε],
                                :S => [:α, :β, :γ, :δ, :ε],
                                )

    wire_collection = Dict() # 82 wires, each a vector of dimension 32
    for (wire_type_B, wire_param_names_B) in wire_types_and_params
        for (wire_type_A, wire_param_names_A) in wire_types_and_params
            #temp_wire_buffer = []
            for wire_param_vals_B in IterTools.product((0:1 for _ in 1:length(wire_param_names_B))...)
                wire_params_B = NamedTuple(zip(wire_param_names_B, wire_param_vals_B))
                for wire_param_vals_A in IterTools.product((0:1 for _ in 1:length(wire_param_names_A))...)
                    wire_params_A = NamedTuple(zip(wire_param_names_A, wire_param_vals_A))
                    wire_collection[(wire_type_A, wire_params_A, wire_type_B, wire_params_B)] = unconditional_extremal_wires(wire_type_A, wire_params_A, wire_type_B, wire_params_B)
                end
            end
        end
    end

    return wire_collection
end





function extremal_wires(wire_types_A::NTuple{2, Symbol}, wire_params_A::NTuple{2, NamedTuple}, wire_types_B::NTuple{2, Symbol}, wire_params_B::NTuple{2, NamedTuple})
    #extremal_wires(::Tuple{Symbol, Symbol}, ::Tuple{@NamedTuple{α::Int64, β::Int64, γ::Int64, δ::Int64, ε::Int64}, @NamedTuple{α::Int64, β::Int64, γ::Int64, δ::Int64, ε::Int64}}, ::Tuple{Symbol, Symbol}, ::Tuple{@NamedTuple{α::Int64, β::Int64, γ::Int64, δ::Int64, ε::Int64}, @NamedTuple{α::Int64, β::Int64, γ::Int64, δ::Int64, ε::Int64}})

    
    @assert all((wire_types_A[i] in [:D, :O, :X, :A, :S]) for i in eachindex(wire_types_A)) "There was given some invalid extremal wire type for A"
    @assert all((wire_types_B[i] in [:D, :O, :X, :A, :S]) for i in eachindex(wire_types_B)) "There was given some invalid extremal wire type for B"

    @assert all(all([x in [0,1] for x in values(wire_params_A[i])]) for i in eachindex(wire_params_A)) "There was given some invalid extremal wire parameters for A"
    @assert all(all([x in [0,1] for x in values(wire_params_B[i])]) for i in eachindex(wire_params_B)) "There was given some invalid extremal wire parameters for B"

    wire_buffer = []
    for (c_wire_types, c_wire_params) in zip((wire_types_A, wire_types_B), (wire_params_A, wire_params_B))
        f1, f2, f3 = zeros(2,2), zeros(2,2), zeros(2,2,2)
        for i in eachindex(c_wire_types)
            wire_type, params = c_wire_types[i], c_wire_params[i]
            
            if wire_type == :D
                f3 .= params[:α]
            elseif wire_type == :O
                f1 .= params[:α]
                f2 .= params[:α]
                if params[:β] == 0
                    for a1 in 0:1
                        f3[i,a1+1,:] .= (a1 ⊻ params[:γ])
                    end
                else #β == 1
                    for a2 in 0:1
                        f3[i,:,a2+1] .= (a2 ⊻ params[:γ])
                    end
                end
            elseif wire_type == :X
                f1 .= params[:α]
                f2 .= params[:β]
                for a2 in 0:1
                    for a1 in 0:1
                        f3[i,a1+1,a2+1] = (a1 ⊻ a2 ⊻ params[:γ])
                    end
                end
            elseif wire_type == :A
                f1 .= params[:α]
                f2 .= params[:β]
                for a2 in 0:1
                    for a1 in 0:1
                        f3[i,a1+1,a2+1] = ((a1 ⊻ params[:γ])*(a2 ⊻ params[:δ]) ⊻ params[:ε])
                    end
                end
            elseif wire_type == :S
                if params[:α] == 0
                    f1 .= params[:β]
                    for a1 in 0:1
                        f2[i,a1+1] = (a1 ⊻ params[:γ])
                    end
                    for a2 in 0:1
                        for a1 in 0:1
                            f3[i,a1+1,a2+1] = ((params[:δ]*a1) ⊻ a2 ⊻ params[:ε])
                        end
                    end
                else #α == 1
                    for a2 in 0:1
                        f2[i,a2+1] = (a2 ⊻ params[:γ])
                    end
                    f2 .= params[:β]
                    for a2 in 0:1
                        for a1 in 0:1
                            f3[i,a1+1,a2+1] = (a1 ⊻ (params[:δ]*a2) ⊻ params[:ε])
                        end
                    end
                end
            else 
                error("Unknown wire type $(wire_type)")
            end
        end 
        push!(wire_buffer, (f1, f2, f3))
    end
    return wirings.functions_to_wires(wire_buffer[1][1], wire_buffer[2][1], wire_buffer[1][2], wire_buffer[2][2], wire_buffer[1][3], wire_buffer[2][3])

end


function collect_all_extremal_wires()
    """Compute all 45,212,176 = 6,724^2 = 82^4 input-dependent extremal wires, each a vector of dimension 32."""

    function recompute_extremal_wires()
        
        wire_types_and_params = Dict(:D => [:α,], 
                                    :O => [:α, :β, :γ], 
                                    :X => [:α, :β, :γ],
                                    :A => [:α, :β, :γ, :δ, :ε],
                                    :S => [:α, :β, :γ, :δ, :ε],
                                    )

        wire_collection = Dict()
        @showprogress desc="Iterating type combinations of B..." for c_wiretype_paramnames_pairs_B in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...) # 2 = 2 inputs = CHSH
            c_wire_types_B, c_wire_param_names_B = Tuple(t[1] for t in c_wiretype_paramnames_pairs_B), Tuple(t[2] for t in c_wiretype_paramnames_pairs_B)
            @showprogress desc="Iterating type combinations of A..." for c_wiretype_paramnames_pairs_A in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...) # 2 = 2 inputs = CHSH
                c_wire_types_A, c_wire_param_names_A = Tuple(t[1] for t in c_wiretype_paramnames_pairs_A), Tuple(t[2] for t in c_wiretype_paramnames_pairs_A)
                
                for c_wire_param_vals_B in Iterators.product((Iterators.product((0:1 for _ in c_wire_param_names_B[i])...) for i in eachindex(c_wire_types_B))...)
                    
                    c_wire_params_B = Tuple(NamedTuple(zip(c_wire_param_names_B[i], c_wire_param_vals_B[i])) for i in eachindex(c_wire_types_B))
                    for c_wire_param_vals_A in Iterators.product((Iterators.product((0:1 for _ in c_wire_param_names_A[i])...) for i in eachindex(c_wire_types_A))...)

                        c_wire_params_A = Tuple(NamedTuple(zip(c_wire_param_names_A[i], c_wire_param_vals_A[i])) for i in eachindex(c_wire_types_A))
                        wire_collection[(c_wire_types_A, c_wire_params_A, c_wire_types_B, c_wire_params_B)] = extremal_wires(c_wire_types_A, c_wire_params_A, c_wire_types_B, c_wire_params_B)
                    end
                end
            end
        end
        return wire_collection
    end

    savedir = mkpath(datadir("ExtremalWires"))
    save_path = datadir("ExtremalWires", "extremal_wires_CHSH.jls")

    if isfile(save_path)
        # File exists, deserialize the data from the file
        println("File existed. Loading extremal wirings from file: $save_path")
        output = deserialize(save_path)
    else
        # File doesn't exist, compute the data and serialize it to a file
        println("File did not exist yet. Computing extremal wirings and saving to file: $save_path")
        output = recompute_extremal_wires()
        open(save_path, "w") do io
            serialize(io, output)
        end
    end
               
    return output
end




#Extremal wire types & parameters (binary):
wire_types_and_params = Dict(:D => [:α,], 
:O => [:α, :β, :γ], 
:X => [:α, :β, :γ],
:A => [:α, :β, :γ, :δ, :ε],
:S => [:α, :β, :γ, :δ, :ε],
)

function extremal_wires_generator()
    return Channel() do ch
        @showprogress "Iterating type combinations of B..." for c_wiretype_paramnames_pairs_B in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...)
            c_wire_types_B, c_wire_param_names_B = Tuple(t[1] for t in c_wiretype_paramnames_pairs_B), Tuple(t[2] for t in c_wiretype_paramnames_pairs_B)
            @showprogress "Iterating type combinations of A..." for c_wiretype_paramnames_pairs_A in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...)
                c_wire_types_A, c_wire_param_names_A = Tuple(t[1] for t in c_wiretype_paramnames_pairs_A), Tuple(t[2] for t in c_wiretype_paramnames_pairs_A)
                for c_wire_param_vals_B in Iterators.product((Iterators.product((0:1 for _ in c_wire_param_names_B[i])...) for i in eachindex(c_wire_types_B))...)
                    c_wire_params_B = Tuple(NamedTuple(zip(c_wire_param_names_B[i], c_wire_param_vals_B[i])) for i in eachindex(c_wire_types_B))
                    for c_wire_param_vals_A in Iterators.product((Iterators.product((0:1 for _ in c_wire_param_names_A[i])...) for i in eachindex(c_wire_types_A))...)
                        c_wire_params_A = Tuple(NamedTuple(zip(c_wire_param_names_A[i], c_wire_param_vals_A[i])) for i in eachindex(c_wire_types_A))
                        
                        put!(ch, (extremal_wires(c_wire_types_A, c_wire_params_A, c_wire_types_B, c_wire_params_B), (c_wire_types_A, c_wire_types_B), (c_wire_params_A, c_wire_params_B)))
                    end
                end
            end
        end
    end
end



# ----- Reduced Extremal Wirings -----

function canonical_extremal_wires_generator()
    return Channel() do ch
        for c_wiretype_paramnames_pairs_B in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...)
            c_wire_types_B, c_wire_param_names_B = Tuple(t[1] for t in c_wiretype_paramnames_pairs_B), Tuple(t[2] for t in c_wiretype_paramnames_pairs_B)
            for c_wiretype_paramnames_pairs_A in Iterators.product((zip(keys(wire_types_and_params), values(wire_types_and_params)) for _ in 1:2)...)
                c_wire_types_A, c_wire_param_names_A = Tuple(t[1] for t in c_wiretype_paramnames_pairs_A), Tuple(t[2] for t in c_wiretype_paramnames_pairs_A)
                
                c_wire_params_B = Tuple(NamedTuple(zip(c_wire_param_names_B[i], Tuple(zeros(Int,length(c_wire_param_names_B[i]))))) for i in eachindex(c_wire_types_B))
                c_wire_params_A = Tuple(NamedTuple(zip(c_wire_param_names_A[i], Tuple(zeros(Int,length(c_wire_param_names_A[i]))))) for i in eachindex(c_wire_types_A))
                
                put!(ch, (extremal_wires(c_wire_types_A, c_wire_params_A, c_wire_types_B, c_wire_params_B), (c_wire_types_A, c_wire_types_B), (c_wire_params_A, c_wire_params_B)))
            end
        end
    end
end



#----- Individual wirings from the literature: -----

function Allcock2009_wires() 
    """x1 = x, x2 = x ⊕ a1 ⊕ 1, a = a1 ⊕ a2 ⊕ 1; 
        y1 = y, y2 = y*b1, b = b1 ⊕ b2 ⊕ 1 """
    
    c_wire_types_A, c_wire_params_A = (:S, :S), Tuple((α=0, β=x, γ=(1 ⊻ x), δ=1, ε=1) for x in 0:1)
    c_wire_types_B, c_wire_params_B = (:X, :S), ((α=0, β=0, γ=1), (α=0, β=1, γ=0, δ=1, ε=1))
    return extremal_wires(c_wire_types_A, c_wire_params_A, c_wire_types_B, c_wire_params_B)
end

            