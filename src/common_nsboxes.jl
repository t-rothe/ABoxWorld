!isdefined(Main, :nsboxes) ? (include("nsboxes.jl"); using .nsboxes) : nothing
using LinearAlgebra, Symbolics

function UniformRandomBox(scenario)
    """Returns a (specific!) NSBox for correlations obtainable from a Uniform Random Box / Maximally Mixed State
    WARNING: This is not a valid generator for a NSBoxFamily, as it has no parameters / keyword arguments for any fixed scenario
    """
    dimX, dimY, dimA, dimB = scenario
    P_abxy_uniform = 1/(dimA*dimB) .* ones(dimA, dimB, dimX, dimY)
    return nsboxes.NSBox(scenario, P_abxy_uniform)
end


function MaxEntangledBoxCHSH()
    """Returns a (specific!) NSBox for correlations obtainable from a Maximally Entangled Quantum State - Singlet (with "optimal" measurements)
    -> Corresponds to eq. (3.11) in "Bell Non-Locality" Book by Scarani 2019. 
    -> Maximally violates the Quantum ("Tsirelson") bound of the CHSH inequality
    WARNING: This is not a valid generator for a NSBoxFamily, as it has no parameters / keyword arguments for any fixed scenario
    """
    scenario = (2,2,2,2)
    P_abxy_maxEntangled = Array{Float64}(undef, 2, 2, 2, 2)
    for a in [-1,1]
        for b in [-1,1]
            for x in 0:1
                for y in 0:1
                    P_abxy_maxEntangled[Int((a+1)/2) + 1,Int((b+1)/2) + 1, x+1, y+1] = 1/4 * (1 + (1/sqrt(2) * a*b*(-1)^(x*y)) ) 
                end
            end
        end
    end
    return nsboxes.NSBox(scenario, P_abxy_maxEntangled)
end

function CanonicalPRBox(d=2)
    """Returns a (specific!) NSBox for the canonical PR box in any (2,2,d,d) scenario (default = CHSH)
    WARNING: This is not a valid generator for a NSBoxFamily, as it has no parameters / keyword arguments for any fixed scenario
    """
    scenario = (2,2,d,d)
    PRBox_abxy = Array{Float64}(undef, d, d, 2, 2)
    #The following is equivalent to:
    #            | 1/k     if (b-a) mod d = x*y
    # P(ab|xy) = |
    #            | 0       otherwise

    A_d = zeros(d,d)
    A_d[1, d] = 1
    for i in 2:d
        A_d[i, i-1] = 1
    end 

    PRBox_abxy[:,:,1,1] = 1/d .* I(d)
    PRBox_abxy[:,:,1,2] = 1/d .* I(d)
    PRBox_abxy[:,:,2,1] = 1/d .* I(d)
    PRBox_abxy[:,:,2,2] = 1/d .* A_d

    #1/d .* [I(d) I(d);  A_d]
    
    return nsboxes.NSBox(scenario, PRBox_abxy)
end

CanonicalPRBox(;d=2) = CanonicalPRBox(d)


# ----------------------------------------------------------------- #
# -------------------  NSBoxFamily Generators ------------------- #
# ----------------------------------------------------------------- #

function getLocalDeterministicBoxGenerator(scenario)
    """Returns a NSBoxFamily generator for the local deterministic boxes in the given scenario
    WARNING: The returned generator has !very different! kind of parameters than the (more efficient) LocalDeterminsitcBoxesCHSH generator, even in the CHSH scenario!
    """
    function LocalDeterminsitcBoxes(;θ_a, θ_b, unsafe=false)
        if !unsafe
            @assert size(θ_a) == (scenario[1], scenario[2])
            @assert size(θ_b) == (scenario[1], scenario[2])
            
            #Check that for constant x and varying y, a is constant and for constant y and varying x, b is constant, i.e. check that all columns of θ_a and all rows of θ_b are equal respectively
            @assert all(θ_a .== θ_a[:,1:1]) && all(θ_b .== θ_b[1:1,:]) "Parameters θ_a and θ_b must be constant in the second and first index respectively; Local Deterministc = not dependent on the other party's input"  
        end

        LocalDetBox = zeros(scenario[3], scenario[4], scenario[1], scenario[2])
        for x in 0:scenario[1]-1 #low index freq
            for y in 0:scenario[2]-1 #high index freq
                LocalDetBox[θ_a[x+1,y+1]+1, θ_b[x+1,y+1]+1, x+1, y+1] = 1.0
            end
        end
        #display(LocalDetBox)
        return nsboxes.NSBox(scenario, LocalDetBox)
    end

    return LocalDeterminsitcBoxes
end

function LocalDeterministicBoxFamily(scenario)
    """Returns a NSBoxFamily for the local deterministic boxes in the given scenario"""
    @variables θ_a[1:scenario[1], 1:scenario[2]], θ_b[1:scenario[1], 1:scenario[2]]
    return nsboxes.NSBoxFamily(scenario=scenario, parameters=(θ_a, θ_b), generator=getLocalDeterministicBoxGenerator(scenario))
end

function LocalDeterministicBoxesCHSH(;α, γ, β, λ, unsafe=false)
    """Generates a local deterministic box with parameters α, β, γ, λ in the CHSH scenario
    WARNING: This generator has !very different! kind of parameters than the more general LocalDeterminsitcBoxes generator obtained by getLocalDeterministicBoxGenerator for the CHSH scenario!
    
    unsafe keyword : avoids checks and should allow Num-type variable input for parameters (i.e. replace boolean checks/operations!)
    """
    
    !unsafe && (@assert α ∈ [0,1] && β ∈ [0,1] && γ ∈ [0,1] && λ ∈ [0,1])
    
    LocalDetBox = Array{Float64}(undef, 2, 2, 2, 2)
    for x in 0:2-1
        for y in 0:2-1
            for a in 0:2-1
                for b in 0:2-1
                    if unsafe
                        LocalDetBox[a+1,b+1,x+1,y+1] = 1.0*(mod(1 + mod(a + (α*x + γ),2), 2) * mod(1 + mod(b + (β*y + λ), 2), 2))
                    else
                        LocalDetBox[a+1,b+1,x+1,y+1] = 1.0*((1 ⊻ (a ⊻ (α*x + γ))) & (1 ⊻ (b ⊻ (β*y + λ))))
                    end
                end
            end
        end
    end
    #display(LocalDetBox)
    return nsboxes.NSBox((2,2,2,2), LocalDetBox; unsafe=unsafe)
end

function LocalDeterministicBoxesCHSH_family() 
    @variables α, β, γ, λ 
    return nsboxes.NSBoxFamily(scenario=(2,2,2,2), parameters=(α, β, γ, λ), generator=LocalDeterministicBoxesCHSH)
end

function PRBoxesCHSH(;μ, ν, σ, unsafe=false)
    """Generates a PR box with parameters μ, ν, σ (Canonical PRBox = 0,0,0) in the CHSH scenario"""
    !unsafe && (@assert μ ∈ [0,1] && ν ∈ [0,1] && σ ∈ [0,1] "Parameters μ, ν, σ must be 0 or 1")
    PRBox = Array{Float64}(undef, 2, 2, 2, 2)
    for a in 0:1
        for b in 0:1
            for x in 0:1
                for y in 0:1
                    PRBox[a+1,b+1,x+1,y+1] = 1/2*(1 - (a ⊻ b) ⊻ ((x*y) ⊻ (μ*x) ⊻ (ν*y) ⊻ σ))
                end
            end
        end
    end
    return nsboxes.NSBox((2,2,2,2), PRBox)
end

function PRBoxesCHSH_family()
    @variables μ, ν, σ
    return nsboxes.NSBoxFamily(scenario=(2,2,2,2), parameters=(μ, ν, σ), generator=PRBoxesCHSH)
end

function IsotropicBoxesCHSH(;ρ, unsafe=false)
    """Generates an isotropic box with parameter p in the CHSH scenario"""
    !unsafe && (@assert 0 <= ρ <= 1 "Parameter p must be between 0 and 1")
    return ρ*CanonicalPRBox() + (1-ρ)*UniformRandomBox((2,2,2,2))
end

function IsotropicBoxesCHSH_family()
    @variables ρ
    return nsboxes.NSBoxFamily(scenario=(2,2,2,2), parameters=(ρ,), generator=IsotropicBoxesCHSH)
end
