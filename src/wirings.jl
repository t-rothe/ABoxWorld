module wirings
    #using Symbolics
    using LinearAlgebra
    using IterTools
    using TensorOperations

    !isdefined(Main, :nsboxes) ? error("First import the nsboxes module") : nothing
    using ..nsboxes

    #export Wiring, in_out_split, direct_boxproduct,convert_nsjoint_to_matrixbox, convert_matrixbox_to_nsjoint, vectorized_boxproduct, tensorized_boxproduct

    global_eps_tol = 1e-10

    struct Wiring
        """Wiring between two bipartite NSBoxes in a symmetric (n,n,d,d) Bell scenerio
    
        Defined through truth tables of f1(x, a2), g1(y,b2), f2(x, a1), g2(y, b1), f3(x, a1, a2) and g3(y, b1, b2) and stored as contiguous 1D array
        Array indices (wires):
        f1(x,a_2) = 2x + a_2
        g1(y,b_2) = 2y + b_2 + 4 
        f2(x,a_1) = 2x + a_1 + 8
        g2(y,b_1) = 2y + b_1 + 12 
        f3(x,a_1,a_2) = 4x + 2a_1 + a_2 + 16
        g3(y,b_1,b_2) = 4y + 2b_1 + b_2 + 24
        """

        scenario::NTuple{4, <:Int}
        wires::Vector{<:Int}

        function Wiring(scenario::NTuple{4, <:Int}, wires::Vector{<:Int})
            @assert scenario == (2,2,2,2) #Allow only CHSH scenario for now
            #@assert scenario[1] == scenario[2]
            #@assert scenario[3] == scenario[4]
            @assert length(wires) == 4*(scenario[1]*scenario[3]) + 2 *(scenario[1]*(scenario[3]^2))
            #Might need for non-CHSH scenario's check that function values in vector are in in- (f1,f2,g1,g2) and output (f3, g3) range
            @assert all(0 .<= wires[1:4*(scenario[1]*scenario[3])] .<= scenario[1] - 1) && all(0 .<= wires[4*(scenario[1]*scenario[3])+1:end] .<= scenario[3] - 1)

            #Check valid wire conditions (ONLY CHSH scenario for now):
            @assert ((wires[1]-wires[2])*(wires[9]-wires[10]) == 0) && ((wires[3]-wires[4])*(wires[11]-wires[12]) == 0) && ((wires[5]-wires[6])*(wires[13]-wires[14]) == 0) && ((wires[7]-wires[8])*(wires[15]-wires[16]) == 0)
            
            new(scenario, wires)
        end
    end

    #==========================================#
    # Utility Functions related to wirings
    #==========================================#

    function functions_to_wires(f1::Matrix{<:Any}, g1::Matrix{<:Any}, f2::Matrix{<:Any}, g2::Matrix{<:Any}, f3::Array{<:Any,3}, g3::Array{<:Any,3})
        # the fi and gj are tensors

        @assert length(f1) == 4 && length(g1) == 4 && length(f2) == 4 && length(g2) == 4 && length(f3) == 8 && length(g3) == 8

        W_vec = zeros(32)
        for x in 0:1
            for a in 0:1
                W_vec[2*x+a+1] = f1[x+1,a+1]
                W_vec[2*x+a+4+1] = g1[x+1,a+1]
                W_vec[2*x+a+8+1] = f2[x+1,a+1]
                W_vec[2*x+a+12+1] = g2[x+1,a+1]
                for a2 in 0:1
                    W_vec[4*x+2*a+a2+16+1] = f3[x+1,a+1,a2+1]
                    W_vec[4*x+2*a+a2+24+1] = g3[x+1,a+1,a2+1]
                end
            end
        end
        return W_vec
    end

    function print_functions_from_wires(W::Vector{<:Any}) 
        #Prints the functions f1, g1, f2, g2, f3, g3 corresponding to the wires W
        
        #TODO: Debug; Found error for extremal wirings; E.g. (:X, :A) with ((1,1,1),(0,0,0,0)) -> Output: f3(:X) = -a1 ⊕ -a2 ⊕ 1.0 while it should be f3(:X) = a1 ⊕ a2 ⊕ 1.0 
        #-> Is an issue with printing, *not* with the (construction of the) wiring itself! (tested)

        # f1
        f1_str0 = "f_1(x,a2) = "
        f1_str = f1_str0
        c = float((W[2+1]-W[0+1])%2)
        if c != 0
            c != 1 && (f1_str *= (string(c)*" "))
            f1_str *= "x"
        end
        c = float((W[1+1]-W[0+1])%2)
        if c != 0
            f1_str != f1_str0 && (f1_str *= " ⊕ ")
            c != 1 && (f1_str *= (string(c)*" "))
            f1_str *= "a2"
        end
        c = float((W[3+1]-W[2+1]-W[1+1]+W[0+1])%2)
        if c !=0
            f1_str != f1_str0 && (f1_str *= " ⊕ ")
            c != 1 && (f1_str *= (string(c)*" "))
            f1_str *= "x·a2"
        end
        c = float((W[0+1])%2)
        if c !=0
            f1_str != f1_str0 && (f1_str *= " ⊕ ")
            f1_str *= (string(c)*" ")
        end

        if f1_str != f1_str0
            println(f1_str)
        else
            f1_str = f1_str0*"0"
            println(f1_str)
        end

        # g1
        g1_str0 = "g_1(y,b2) = "
        g1_str = g1_str0
        c = float((W[6+1]-W[4+1])%2)
        if c != 0
            c != 1 && (g1_str *= (string(c)*" "))
            g1_str *= "y"
        end
        c = float((W[5+1]-W[4+1])%2)
        if c != 0
            g1_str != g1_str0 && (g1_str *= " ⊕ ")
            c != 1 && (g1_str *= (string(c)*" "))
            g1_str *= "b2"
        end
        c = float((W[7+1]-W[6+1]-W[5+1]+W[4+1])%2)
        if c !=0
            g1_str != g1_str0 && (g1_str *= " ⊕ ")
            c != 1 && (g1_str *= string(c)*" ")
            g1_str *= "y·b2"
        end
        c = float((W[4+1])%2)
        if c !=0
            g1_str != g1_str0 && (g1_str *= " ⊕ ")
            g1_str *= (string(c)*" ")
        end

        if g1_str != g1_str0
            println(g1_str)
        else
            g1_str = g1_str0*"0"
            println(g1_str)
        end

        # f2
        f2_str0 = "f_2(x,a1) = "
        f2_str = f2_str0
        c = float((W[10+1]-W[8+1])%2)
        if c != 0
            c != 1 && (f2_str *= (string(c)*" "))
            f2_str *= "x"
        end
        c = float((W[9+1]-W[8+1])%2)
        if c != 0
            f2_str != f2_str0 && (f2_str *= " ⊕ ")
            c != 1 && (f2_str *= (string(c)*" "))
            f2_str *= "a1"
        end
        c = float((W[11+1]-W[10+1]-W[9+1]+W[8+1])%2)
        if c !=0
            f2_str != f2_str0 && (f2_str *= " ⊕ ")
            c != 1 && (f2_str *= (string(c)*" "))
            f2_str *= "x·a1"
        end
        c = float((W[8+1])%2)
        if c !=0
            f2_str != f2_str0 && (f2_str *= " ⊕ ")
            f2_str *= (string(c)*" ")
        end
        
        if f2_str != f2_str0
            println(f2_str)
        else
            f2_str = f2_str0*"0"
            println(f2_str)
        end

        # g2
        g2_str0 = "g_2(y,b1) = "
        g2_str = g2_str0
        c = float((W[14+1]-W[12+1])%2)
        if c != 0
            c != 1 && (g2_str *= (string(c)*" "))
            g2_str *= "y"
        end
        c = float((W[13+1]-W[12+1])%2)
        if c != 0
            g2_str != g2_str0 && (g2_str *= " ⊕ ")
            c != 1 && (g2_str *= (string(c)*" "))
            g2_str *= "b1"
        end
        c = float((W[15+1]-W[14+1]-W[13+1]+W[12+1])%2)
        if c !=0
            g2_str != g2_str0 && (g2_str *= " ⊕ ")
            c != 1 && (g2_str *= (string(c)*" "))
            g2_str *= "y·b1"
        end
        c = float((W[12+1])%2)
        if c !=0
            g2_str != g2_str0 && (g2_str *= " ⊕ ")
            g2_str *= (string(c)*" ")
        end
        
        if g2_str != g2_str0
            println(g2_str)
        else
            g2_str = g2_str0*"0"
            println(g2_str)
        end

        # f3
        f3_str0 = "f_3(x,a1,a2) = "
        f3_str = f3_str0
        c = float((W[20+1]-W[16+1])%2)
        if c != 0
            c != 1 && (f3_str *= (string(c)*" "))
            f3_str *= "x"
        end
        c = float((W[18+1]-W[16+1])%2)
        if c != 0
            f3_str != f3_str0 && (f3_str *= " ⊕ ")
            c != 1 && (f3_str *= (string(c)*" "))
            f3_str *= "a1"
        end
        c = float((W[17+1]-W[16+1])%2)
        if c != 0
            f3_str != f3_str0 && (f3_str *= " ⊕ ")
            c != 1 && (f3_str *= (string(c)*" "))
            f3_str *= "a2"
        end
        c = float((W[21+1]-W[20+1]-W[18+1]+W[16+1])%2)
        if c != 0
            f3_str != f3_str0 && (f3_str *= " ⊕ ")
            c != 1 && (f3_str *= (string(c)*" "))
            f3_str *= "x·a1"
        end
        c = float((W[22+1]-W[20+1]-W[17+1]+W[16+1])%2)
        if c != 0
            f3_str != f3_str0 && (f3_str *= " ⊕ ")
            c != 1 && (f3_str *= (string(c)*" "))
            f3_str *= "x·a2"
        end
        c = float((W[19+1]-W[18+1]-W[17+1]+W[16+1])%2)
        if c != 0
            f3_str != f3_str0 && (f3_str *= " ⊕ ")
            c != 1 && (f3_str *= (string(c)*" "))
            f3_str *= "a1·a2"
        end
        c = float((W[23+1] - W[21+1] - W[22+1]+W[20+1] - W[19+1]+W[18+1]+W[17+1] - W[16+1])%2)
        if c != 0
            f3_str != f3_str0 && (f3_str *= " ⊕ ")
            c != 1 && (f3_str *= (string(c)*" "))
            f3_str *= "x·a1·a2"
        end
        c = float((W[16+1])%2)
        if c !=0
            f3_str != f3_str0 && (f3_str *= " ⊕ ")
            f3_str *= (string(c)*" ")
        end
        
        if f3_str != f3_str0
            println(f3_str)
        else
            f3_str = f3_str0*"0"
            println(f3_str)
        end

        # g3
        g3_str0 = "g_3(y,b1,b2) = "
        g3_str = g3_str0
        c = float((W[28+1]-W[24+1])%2)
        if c != 0
            c != 1 && (g3_str *= (string(c)*" "))
            g3_str *= "y"
        end
        c = float((W[26+1]-W[24+1])%2)
        if c != 0
            g3_str != g3_str0 && (g3_str *= " ⊕ ")
            c != 1 && (g3_str *= (string(c)*" "))
            g3_str *= "b1"
        end
        c = float((W[25+1]-W[24+1])%2)
        if c != 0
            g3_str != g3_str0 && (g3_str *= " ⊕ ")
            c != 1 && (g3_str *= (string(c)*" "))
            g3_str *= "b2"
        end
        c = float((W[29+1]-W[28+1]-W[26+1]+W[24+1])%2)
        if c != 0
            g3_str != g3_str0 && (g3_str *= " ⊕ ")
            c != 1 && (g3_str *= (string(c)*" "))
            g3_str *= "y·b1"
        end
        c = float((W[30+1]-W[28+1]-W[25+1]+W[24+1])%2)
        if c != 0
            g3_str != g3_str0 && (g3_str *= " ⊕ ")
            c != 1 && (g3_str *= (string(c)*" "))
            g3_str *= "y·b2"
        end
        c = float((W[27+1]-W[26+1]-W[25+1]+W[24+1])%2)
        if c != 0
            g3_str != g3_str0 && (g3_str *= " ⊕ ")
            c != 1 && (g3_str *= (string(c)*" "))
            g3_str *= "b1·b2"
        end
        c = float((W[31+1] - W[29+1] - W[30+1]+W[28+1] - W[27+1]+W[26+1]+W[25+1] - W[24+1])%2)
        if c != 0
            g3_str != g3_str0 && (g3_str *= " ⊕ ")
            c != 1 && (g3_str *= (string(c)*" "))
            g3_str *= "y·b1·b2"
        end
        c = float((W[24+1])%2)
        if c != 0
            g3_str != g3_str0 && (g3_str *= " ⊕ ")
            g3_str *= (string(c)*" ")
        end

        if g3_str != g3_str0
            println(g3_str)
        else
            g3_str = g3_str0*"0"
            println(g3_str)
        end
    end

    function print_functions_from_wires(W::Matrix{<:Any}) #Assuming 32xn matrix
        if size(W)[2] > 1
            warning("Print-utility for wirings recieved a matrix with $(size(W)[2]) wirings. \n We only print the first one and ignore the rest")
            W = W[:,1]
        end
        print_functions_from_wiring(W)
    end

    print_functions_from_wiring(W::Wiring) = print_functions_from_wires(W.wires)


    project_to_deterministic_wires(W::Union{Matrix{<:Real}, Vector{<:Real}}) = round.(W)
    

    #==========================================#
    # Methods to apply a wiring between boxes = boxproduct
    # Currently choose between direct formula, a vectorized version and a tensorized version
    #==========================================#

    function in_out_split(w::Vector{<:Any})        
        in_dim, out_dim = 2, 2 
        in_vec_length, out_vec_length = 4*(in_dim*out_dim), 2*(in_dim*(out_dim^2))

        #Splitting the vector into input and output wires
        in_vec, out_vec = reshape(w[1:in_vec_length], (out_dim, in_dim, 4)), reshape(w[in_vec_length+1:end], (out_dim, out_dim, in_dim, 2)) #Order: (out idx, in idx, func idx) and (out idx 2, out idx 1, in idx, func iddx)
        return permutedims(in_vec, (3,2,1)), permutedims(out_vec, (4,3,2,1)) #Order: (func idx, in idx, out idx)
    end

    in_out_split(w::Wiring) = in_out_split(w.wires)

    # [1] The direct way to calculate the box product by the explicit formula:
    #------------------------------------------#

    function direct_boxproduct(w::Wiring, nsjoint1::Array{Float64,4}, nsjoint2::Array{Float64,4})
        in_vec, out_vec = in_out_split(w)
        reverseP, reverseQ = permutedims(nsjoint1, (3,4,1,2)), permutedims(nsjoint2, (3,4,1,2)) #Order s.t. inputs first and outputs second
        box_prod = zeros(Float64, size(reverseP))
        for (a1, a2, b1, b2) in Iterators.product(1:w.scenario[3], 1:w.scenario[3], 1:w.scenario[3], 1:w.scenario[3])
            for (a,b) in Iterators.product(1:w.scenario[3], 1:w.scenario[3])
                for (x,y) in Iterators.product(1:w.scenario[1], 1:w.scenario[1])
                    #NOTE 1-indexing in the iterator specifications and following exprresion
                    f1, g1, f2, g2, f3, g3 = in_vec[1,x,a2], in_vec[2,y,b2], in_vec[3,x,a1], in_vec[4,y,b1], out_vec[1,x,a1,a2], out_vec[2,y,b1,b2]
                    box_prod[x,y,a,b] += ( 
                        ((1-f1)*(1-g1)*reverseP[1,1,a1,b1] + (1-f1)*g1*reverseP[1,2,a1,b1] + 
                        f1*(1-g1)*reverseP[2,1,a1,b1] + f1*g1*reverseP[2,2,a1,b1]) *
                        ((1-f2)*(1-g2)*reverseQ[1,1,a2,b2] + (1-f2)*g2*reverseQ[1,2, a2,b2] +
                        f2*(1-g2)*reverseQ[2,1,a2,b2] + f2*g2*reverseQ[2,2,a2,b2]) *
                        ((1-f3)*(a==1) + f3*(a==2)) * ((1-g3)*(b==1) + g3*(b==2))
                                        )
                end
            end
        end
        return permutedims(box_prod, (3,4,1,2)) #Change order back to outputs first and inputs second
    end
    
    # [2] The vectorized way to calculate the box product:
    #------------------------------------------#
    
    function convert_nsjoint_to_matrixbox(nsjoint::Array{Float64,4}) #Only for CHSH scenario

        matrixbox = Matrix{Float64}(undef, 4, 4)
        for (i, in_pair) in enumerate(Iterators.product(1:2, 1:2))
            for (j, out_pair) in enumerate(Iterators.product(1:2, 1:2))
                matrixbox[i,j] = nsjoint[reverse(out_pair)...,reverse(in_pair)...] #Reverse needed to iterate B's input/output indices faster than A's (see definition of matrixbox)
            end
        end
        return matrixbox
    end
    
    function convert_matrixbox_to_nsjoint(matrixbox::Matrix{Float64}) #Only for CHSH scenario
        fulljoint = zeros(2, 2, 2, 2)
        for (i, in_pair) in enumerate(Iterators.product(1:2, 1:2))
            for (j, out_pair) in enumerate(Iterators.product(1:2, 1:2))
                fulljoint[out_pair...,in_pair...] = matrixbox[i,j]
            end
        end
        
        return fulljoint
    end

    sum_coeff(M::Array) = ones(size(M)[1])'*M*ones(size(M)[1])

    F_mat = transpose([-1 -1  0  0 -1 -1  0  0  1  1  0  0  1  1  0  0
                        0  0 -1 -1  0  0 -1 -1  0  0  1  1  0  0  1  1
                        1  1  0  0  1  1  0  0  0  0  0  0  0  0  0  0
                        0  0  1  1  0  0  1  1  0  0  0  0  0  0  0  0 ])

    G_mat = transpose([-1  0 -1  0  1  0  1  0 -1  0 -1  0  1  0  1  0
                        0 -1  0 -1  0  1  0  1  0 -1  0 -1  0  1  0  1
                        1  0  1  0  0  0  0  0  1  0  1  0  0  0  0  0
                        0  1  0  1  0  0  0  0  0  1  0  1  0  0  0  0 ])

    F_tilde_mat = transpose([1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0
                            0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0
                            0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0
                            0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 ])

    G_tilde_mat = transpose([1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0
                            0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0
                            0 0 0 0 1 0 1 0 0 0 0 0 1 0 1 0
                            0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 1])

    function vectorized_boxproduct(w::Wiring, matrixbox1::Matrix{Float64}, matrixbox2::Matrix{Float64})
        box_prod = Array{Float64}(undef,  w.scenario[3], w.scenario[4], w.scenario[1], w.scenario[2])
        
        for (x,y) in Iterators.product(0:w.scenario[1]-1, 0:w.scenario[1]-1)
            term1 = (reshape((F_mat * [w.wires[(2*x)+0+1],w.wires[(2*x)+1+1],1,1]) .* (G_mat * [w.wires[(2*y)+0+4+1],w.wires[(2*y)+1+4+1],1,1]), (4,4)) *matrixbox1)
            term2 = (reshape((F_mat * [w.wires[(2*x)+0+8+1],w.wires[(2*x)+1+8+1],1,1]) .* (G_mat * [w.wires[(2*y)+0+12+1],w.wires[(2*y)+1+12+1],1,1]), (4,4)) *matrixbox2)
                
            for (a,b) in Iterators.product(0:w.scenario[3]-1, 0:w.scenario[3]-1)
                term3f = ((1-a) .* ones(Int, 4,4)) .- (((-1)^a) .* reshape(F_tilde_mat * [w.wires[(4*x)+(2*0)+0+16+1],w.wires[(4*x)+(2*0)+1+16+1],w.wires[(4*x)+(2*1)+0+16+1],w.wires[(4*x)+(2*1)+1+16+1]], (4,4)))
                term3g = ((1-b) .* ones(Int, 4,4)) .- (((-1)^b) .* reshape(G_tilde_mat * [w.wires[(4*y)+(2*0)+0+24+1],w.wires[(4*y)+(2*0)+1+24+1],w.wires[(4*y)+(2*1)+0+24+1],w.wires[(4*y)+(2*1)+1+24+1]], (4,4)))
                    
                box_prod[a+1, b+1, x+1, y+1] = sum_coeff(term1 .* transpose(term2) .* term3f .* term3g)
            end
        end
        return box_prod #Joint probability tensor P[a,b,x,y]
    end


    # [3] The tensorized way to calculate the box product:
    #------------------------------------------#

    function update_nb_columns!(new_nb_columns::Int)#
        global nb_columns = new_nb_columns

        global A1 = zeros(2, 4, 4, 32)
        for x in 0:1
            for j in 0:3
                
                sign = 1
                if j >= 2
                    sign=-1
                end

                A1[x+1, 0+1, j+1, 0+1] = sign*(x-1)
                A1[x+1, 1+1, j+1, 0+1] = sign*(x-1)
                A1[x+1, 2+1, j+1, 1+1] = sign*(x-1)
                A1[x+1, 3+1, j+1, 1+1] = sign*(x-1)
                
                A1[x+1, 0+1, j+1, 2+1] = sign*(-x)
                A1[x+1, 1+1, j+1, 2+1] = sign*(-x)
                A1[x+1, 2+1, j+1, 3+1] = sign*(-x)
                A1[x+1, 3+1, j+1, 3+1] = sign*(-x)
            end
        end

        # A2 is a 2x4x4xn tensor
        global A2 = zeros(2, 4, 4, nb_columns)
        for x in 0:1
            for i in 0:3
                for k in 0:3
                    for α in 0:nb_columns-1
                        if k<=1
                            A2[x+1, i+1, k+1, :] .= 1 #TODO: Might need to adapt this to get same behavior as in Torch
                        end
                    end
                end
            end
        end

        # A3 is a 2x4x4x32-tensor
        global A3 = zeros(2, 4, 4, 32)
        for y in 0:1
            for j in 0:3
                
                sign = 1
                if j==1 || j==3
                    sign = -1
                end

                A3[y+1, 0+1, j+1, 0+4+1] = sign*(y-1)
                A3[y+1, 2+1, j+1, 0+4+1] = sign*(y-1)
                A3[y+1, 1+1, j+1, 1+4+1] = sign*(y-1)
                A3[y+1, 3+1, j+1, 1+4+1] = sign*(y-1)
                
                A3[y+1, 0+1, j+1, 2+4+1] = sign*(-y)
                A3[y+1, 2+1, j+1, 2+4+1] = sign*(-y)
                A3[y+1, 1+1, j+1, 3+4+1] = sign*(-y)
                A3[y+1, 3+1, j+1, 3+4+1] = sign*(-y)
            end
        end

        # A4 is a 2x4x4xn-tensor
        global A4 = zeros(2, 4, 4, nb_columns)
        for y in 0:1
            for i in 0:3
                for k in 0:3
                    for α in 0:nb_columns-1
                        if k==0 || k==2
                            A4[y+1, i+1, k+1, :] .= 1
                        end
                    end
                end
            end
        end
        

        # B1 is a 2x4x4x32-tensor
        global B1 = zeros( 2, 4, 4, 32)
        for x in 0:1
            for l in 0:3
                
                sign = 1
                if l>=2
                    sign=-1
                end

                B1[x+1, 0+1, l+1, 0+8+1] = sign * (x-1)
                B1[x+1, 1+1, l+1, 0+8+1] = sign * (x-1)
                B1[x+1, 2+1, l+1, 1+8+1] = sign * (x-1)
                B1[x+1, 3+1, l+1, 1+8+1] = sign * (x-1)

                B1[x+1, 0+1, l+1, 2+8+1] = sign * (-x)
                B1[x+1, 1+1, l+1, 2+8+1] = sign * (-x)
                B1[x+1, 2+1, l+1, 3+8+1] = sign * (-x)
                B1[x+1, 3+1, l+1, 3+8+1] = sign * (-x)
            end
        end

        # B2 is equal to A2
        global B2 = copy(A2)

        # B3 is a 2x4x4x32-tensor
        global B3 = zeros( 2, 4, 4, 32)
        for y in 0:1
            for l in 0:3
                
                sign=1
                if l==1 || l==3
                    sign=-1
                end

                B3[y+1, 0+1, l+1, 0+12+1] = sign * (y-1)
                B3[y+1, 2+1, l+1, 0+12+1] = sign * (y-1)
                B3[y+1, 1+1, l+1, 1+12+1] = sign * (y-1)
                B3[y+1, 3+1, l+1, 1+12+1] = sign * (y-1)
                
                B3[y+1, 0+1, l+1, 2+12+1] = sign * (-y)
                B3[y+1, 2+1, l+1, 2+12+1] = sign * (-y)
                B3[y+1, 1+1, l+1, 3+12+1] = sign * (-y)
                B3[y+1, 3+1, l+1, 3+12+1] = sign * (-y)
            end
        end

        # B4 is equal to A4
        global B4 = copy(A4)

        # C1 is a 2x2x4x4x32-tensor
        global C1 = zeros( 2, 2, 4, 4, 32)
        for a in 0:1
            for x in 0:1
                for j in 0:3
                    if j<=1
                        C1[a+1, x+1, 0+1, j+1, 0+16+1] = -(1-x) * (-1)^a
                        C1[a+1, x+1, 1+1, j+1, 0+16+1] = -(1-x) * (-1)^a
                        C1[a+1, x+1, 2+1, j+1, 1+16+1] = -(1-x) * (-1)^a
                        C1[a+1, x+1, 3+1, j+1, 1+16+1] = -(1-x) * (-1)^a
                        
                        C1[a+1, x+1, 0+1, j+1, 4+16+1] = -(x) * (-1)^a
                        C1[a+1, x+1, 1+1, j+1, 4+16+1] = -(x) * (-1)^a
                        C1[a+1, x+1, 2+1, j+1, 5+16+1] = -(x) * (-1)^a
                        C1[a+1, x+1, 3+1, j+1, 5+16+1] = -(x) * (-1)^a
                    end

                    if j>=2
                        C1[a+1, x+1, 0+1, j+1, 0+18+1] = -(1-x) * (-1)^a
                        C1[a+1, x+1, 1+1, j+1, 0+18+1] = -(1-x) * (-1)^a
                        C1[a+1, x+1, 2+1, j+1, 1+18+1] = -(1-x) * (-1)^a
                        C1[a+1, x+1, 3+1, j+1, 1+18+1] = -(1-x) * (-1)^a
                        
                        C1[a+1, x+1, 0+1, j+1, 4+18+1] = -(x) * (-1)^a
                        C1[a+1, x+1, 1+1, j+1, 4+18+1] = -(x) * (-1)^a
                        C1[a+1, x+1, 2+1, j+1, 5+18+1] = -(x) * (-1)^a
                        C1[a+1, x+1, 3+1, j+1, 5+18+1] = -(x) * (-1)^a
                    end
                end
            end
        end

        # C2 is a 2x2x4x4xnb_columns-tensor
        global C2 = zeros(2, 2, 4, 4, nb_columns)
        for x in 0:1
            for i in 0:3
                for j in 0:3
                    for α in 0:nb_columns-1
                        C2[0+1, x+1, i+1, j+1, :] .= 1
                    end
                end
            end
        end

        global D1 = zeros( 2, 2, 4, 4, 32 )
        for b in 0:1
            for y in 0:1
                for j in 0:3
                    if j==0 || j==2
                        D1[b+1, y+1, 0+1, j+1, 0+24+1] = -(1-y) * (-1)^b
                        D1[b+1, y+1, 2+1, j+1, 0+24+1] = -(1-y) * (-1)^b
                        D1[b+1, y+1, 1+1, j+1, 0+24+1] = -(1-y) * (-1)^b
                        D1[b+1, y+1, 3+1, j+1, 1+24+1] = -(1-y) * (-1)^b
                        
                        D1[b+1, y+1, 0+1, j+1, 4+24+1] = -(y) * (-1)^b
                        D1[b+1, y+1, 2+1, j+1, 4+24+1] = -(y) * (-1)^b
                        D1[b+1, y+1, 1+1, j+1, 5+24+1] = -(y) * (-1)^b
                        D1[b+1, y+1, 3+1, j+1, 5+24+1] = -(y) * (-1)^b
                    end

                    if j==1 || j==3
                        D1[b+1, y+1, 0+1, j+1, 0+26+1] = -(1-y) * (-1)^b
                        D1[b+1, y+1, 2+1, j+1, 0+26+1] = -(1-y) * (-1)^b
                        D1[b+1, y+1, 1+1, j+1, 1+26+1] = -(1-y) * (-1)^b
                        D1[b+1, y+1, 3+1, j+1, 1+26+1] = -(1-y) * (-1)^b
                        
                        D1[b+1, y+1, 0+1, j+1, 4+26+1] = -(y) * (-1)^b
                        D1[b+1, y+1, 2+1, j+1, 4+26+1] = -(y) * (-1)^b
                        D1[b+1, y+1, 1+1, j+1, 5+26+1] = -(y) * (-1)^b
                        D1[b+1, y+1, 3+1, j+1, 5+26+1] = -(y) * (-1)^b
                    end
                end
            end
        end

        # D2 is a 2x2x4x4xnb_columns-tensor
        global D2 = zeros( 2, 2, 4, 4, nb_columns)
        for y in 0:1
            for i in 0:3
                for j in 0:3
                    for α in 0:nb_columns-1
                        D2[0+1, y+1, i+1, j+1, :] .= 1
                    end
                end
            end
        end
    end

    update_nb_columns!(Int(1e0))

    function A(W::Matrix{<:Real})    # W is a 32xn matrix, A1 is a 2x4x4x32 tensor, A2 is a 2x4x4xn tensor
        #raise Exception(str(A1.dtype), str(W.dtype), str(A2.dtype))
        @tensor T1[i,j,k,l] := A1[i,j,k,m] * W[m,l] + A2[i,j,k,l] 
        T3 = permutedims(repeat(reshape(T1, 1, size(T1)...),2,1,1,1,1), (2,1,3,4,5))
        @tensor S1[i,j,k,l] := A3[i,j,k,m] * W[m,l] + A4[i,j,k,l]
        S2 = repeat(reshape(S1, 1, size(S1)...), 2, 1, 1, 1, 1) #2x2x4x4xn
        #R = permutedims(T3 .* S2, (1,2,5,4,3)) #2x2xnx4x4
        #return permutedims(R, (1,2,3,5,4)) # the output is a 2x2xnx4x4 tensor
        return permutedims(T3 .* S2, (1,2,5,3,4))
    end

    function B(W::Matrix{<:Real})    # W is a 32xn matrix
        @tensor T1[i,j,k,l] := B1[i,j,k,m] * W[m,l] + B2[i,j,k,l]
        T3 = permutedims(repeat(reshape(T1, 1, size(T1)...),2,1,1,1,1), (2,1,3,4,5))
        @tensor S1[i,j,k,l] := B3[i,j,k,m] * W[m,l] + B4[i,j,k,l]
        S2 = repeat(reshape(S1, 1, size(S1)...), 2,1,1,1,1)
        #R = permutedims(T3 .* S2, (1,2,5,4,3))
        #return permutedims(R, (1,2,3,5,4))
        return permutedims(T3 .* S2, (1,2,5,3,4))
        # the output is a 2x2xnx4x4 tensor
    end


    function C(W::Matrix{<:Real})    # W is a 32xn matrix
        @tensor T1[i,j,k,l,m] := C1[i,j,k,l,q] * W[q,m] + C2[i,j,k,l,m]
        #R1 = permutedims(T2, (2,1,3,4,5,6,7))
        #R2 = permutedims(R1, (4,2,3,1,5,6,7))
        #R3 = permutedims(R2, (3,2,1,4,5,6,7))
        #R4 = permutedims(R3, (1,2,3,4,7,6,5))
        #R5 = permutedims(R4, (1,2,3,4,5,7,6))
        #return R5 #TODO: Might replace by single permutedims call
        return permutedims(repeat(reshape(T1, 1, 1, size(T1)...), 2,2,1,1,1,1,1), (3,1,4,2,7,5,6))
        # the output is a 2x2x2x2xnx4x4 tensor
    end

    function D(W::Matrix{<:Real})    # W is a 32xn matrix
        @tensor T1[i,j,k,l,m] := D1[i,j,k,l,q] * W[q,m] + D2[i,j,k,l,m]
        #R = permutedims(T2, (1,3,2,4,5,6,7))
        #R = permutedims(R, (1,2,3,4,7,6,5))
        #R = permutedims(R, (1,2,3,4,5,7,6))
        #return R # the output is a 2x2x2x2xnx4x4 tensor
        return permutedims(repeat(reshape(T1, 1, 1, size(T1)...), 2,2,1,1,1,1,1), (1,3,2,4,7,5,6))
    end


    function tensorized_boxproduct(w::Matrix{<:Real}, matrixbox1::Matrix{Float64}, matrixbox2::Matrix{Float64}) # w is a 32xn tensor, matrixbox1 and matrixbox2 are 4x4 matrices
        @tensor T1[i,j,k,l,m] := A(w)[i,j,k,l,q] * matrixbox1[q,m] # green term; A(w) is a 2x2xnx4x4 tensor
        @tensor T2[i,j,k,l,m] := B(w)[i,j,k,l,q] * matrixbox2[q,m] # blue term; B(w) is a 2x2xnx4x4 tensor
        T4 = repeat(reshape(T1 .* permutedims(T2, (1,2,3,5,4)), 1,1,size(T1)...), 2,2,1,1,1,1,1) .* C(w) .* D(w)  # the big bracket; C(w) and D(w) are 2x2x2x2xnx4x4 tensors
        return @tensor R[i,j,k,l,m] := T4[i,j,k,l,m,s,t] * ones(4,4)[s,t]
        # the output is a 2x2x2x2xn tensor (for n wirings)
    end

    tensorized_boxproduct(w::Matrix{<:Real}, nsjoint1::Array{Float64,4}, nsjoint2::Array{Float64,4}) = tensorized_boxproduct(w, convert_nsjoint_to_matrixbox(nsjoint1), convert_nsjoint_to_matrixbox(nsjoint2))

    reduc_tensorized_boxproduct(w::Matrix{<:Real}, matrixbox1::Matrix{Float64}, matrixbox2::Matrix{Float64}) = tensorized_boxproduct(w, matrixbox1, matrixbox2)[:,:,:,:,1] #Get a single product-box by assuming a single wiring given
    reduc_tensorized_boxproduct(w::Vector{<:Real}, matrixbox1::Matrix{Float64}, matrixbox2::Matrix{Float64}) = reduc_tensorized_boxproduct(reshape(w, 32, 1), matrixbox1, matrixbox2)
    reduc_tensorized_boxproduct(w::Matrix{<:Real}, nsjoint1::Array{Float64,4}, nsjoint2::Array{Float64,4}) = reduc_tensorized_boxproduct(w, convert_nsjoint_to_matrixbox(nsjoint1), convert_nsjoint_to_matrixbox(nsjoint2))
    reduc_tensorized_boxproduct(w::Vector{<:Real}, nsjoint1::Array{Float64,4}, nsjoint2::Array{Float64,4}) = reduc_tensorized_boxproduct(reshape(w, 32, 1), convert_nsjoint_to_matrixbox(nsjoint1), convert_nsjoint_to_matrixbox(nsjoint2))

    #TODO: Make tensorized boxproduct the default for multiple parralell wirings, and vectorized boxproduct the default for single (vector) wirings

    #Base.:*(box1::nsboxes.NSBox, w::Wiring, box2::nsboxes.NSBox) = wire(w, box1, box2) #Conveninece method for wiring two boxes, e.g. box1 * w * box2
    

    #--------
    # Next: The projection of any (32,n) matrix of binary elements to the nearest valid wires-matrix
    #--------

    M1 = Matrix{Float64}(I, 32, 32)
    M1[1,1]=0.5
    M1[1,2]=0.5
    M1[2,1]=0.5
    M1[2,2]=0.5

    M2 = Matrix{Float64}(I, 32, 32)
    M2[9,9]=0.5
    M2[9,10]=0.5
    M2[10,9]=0.5
    M2[10,10]=0.5

    M3 = Matrix{Float64}(I, 32, 32)
    M3[3,3]=0.5
    M3[3,4]=0.5
    M3[4,3]=0.5
    M3[4,4]=0.5

    M4 = Matrix{Float64}(I, 32, 32)
    M4[11,11]=0.5
    M4[11,12]=0.5
    M4[12,11]=0.5
    M4[12,12]=0.5

    M5 = Matrix{Float64}(I, 32, 32)
    M5[5,5]=0.5
    M5[5,6]=0.5
    M5[6,5]=0.5
    M5[6,6]=0.5

    M6 = Matrix{Float64}(I, 32, 32)
    M6[13,13]=0.5
    M6[13,14]=0.5
    M6[14,13]=0.5
    M6[14,14]=0.5

    M7 = Matrix{Float64}(I, 32, 32)
    M7[7,7]=0.5
    M7[7,8]=0.5
    M7[8,7]=0.5
    M7[8,8]=0.5

    M8 = Matrix{Float64}(I, 32, 32)
    M8[15,15]=0.5
    M8[15,16]=0.5
    M8[16,15]=0.5
    M8[16,16]=0.5

    function projected_wiring(W::Matrix{<:Real})  # W is a 32xn tensor
        W = max.(W, zeros(size(W)...))  # it outputs the element-wise maximum
        W = max.(W, ones(size(W)...))   # similarly for minimum

        T1 = (abs.(W[1:1,:] - W[2:2,:]) .<= abs.(W[9:9, :] - W[10:10, :]))
        W = (T1 .* (M1*W)) + ((.!T1) .* (M2*W))

        T2 = (abs.(W[3:3,:] - W[4:4,:]) .<= abs.(W[11:11, :] - W[12:12, :]))
        W = (T2 .* (M3*W)) + ((.!T2) .* (M4*W))

        T3 = (abs.(W[5:5,:] - W[6:6,:]) .<= abs.(W[13:13, :] - W[14:14, :]))
        W = (T3 .* (M5*W)) + ((.!T3) .* (M6*W))

        T4 = (abs.(W[7:7,:] - W[8:8,:]) .<= abs.(W[15:15, :] - W[16:16, :]))
        W = (T4 .* (M7*W)) + ((.!T4) .* (M8*W))

        return W
    end



end