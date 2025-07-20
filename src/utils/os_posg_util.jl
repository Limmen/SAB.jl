using Distributions

function sample_next_state(T::Array{Float64,4}, s_idx::Int, a1_idx::Int, a2_idx::Int)::Int
    return rand(Categorical(T[a1_idx, a2_idx, s_idx, :]))
end

function sample_initial_state(b1::Vector{Float64})::Int
    return rand(Categorical(b1))
end

function sample_next_observation(Z::Array{Float64,4}, s_prime_idx::Int, a1_idx::Int, a2_idx::Int)::Int
    rand(Categorical(Z[a1_idx, a2_idx, s_prime_idx, :]))
end

function bayes_filter(s_prime::Int, o_idx::Int, ai_idx::Int, b::Vector{Float64}, pi2::Matrix{Float64}, num_states::Int,
    num_a2::Int, Z::Array{Float64,4}, T::Array{Float64,4})::Float64    
    norm = sum(
        b[s] * pi2[s, a2] * T[ai_idx, a2, s, s_prime_1] * Z[ai_idx, a2, s_prime_1, o_idx]
        for s in 1:num_states, a2 in 1:num_a2, s_prime_1 in 1:num_states
    )
    if isapprox(norm, 0)
        return 0.0
    end
    temp = sum(
        b[s] * pi2[s, a2] * T[ai_idx, a2, s, s_prime] * Z[ai_idx, a2, s_prime, o_idx]
        for s in 1:num_states, a2 in 1:num_a2
    )
    b_prime_s_prime = temp / norm
    if round(b_prime_s_prime, digits=2) > 1
        @warn "Belief b'(s') > 1: $b_prime_s_prime, a1:$ai_idx, s_prime:$s_prime, o:$o_idx"
    end
    @assert round(b_prime_s_prime, digits=2) <= 1 "Belief cannot exceed 1."
    return b_prime_s_prime
end

function next_belief(o_idx::Int, a1_idx::Int, b::Vector{Float64}, pi2::Matrix{Float64}, num_states::Int, num_a2::Int,
    Z::Array{Float64,4}, T::Array{Float64,4})::Vector{Float64}    
    b_prime = [bayes_filter(s_prime, o_idx, a1_idx, b, pi2, num_states, num_a2, Z, T) for s_prime in 1:num_states]
    s = sum(b_prime)
    if isapprox(s, 0.0, atol=1e-8)
        return ones(num_states) ./ num_states
    end
    @assert isapprox(s, 1.0, atol=0.01) "The new belief vector must sum to 1. Sum = $s"
    return b_prime
end

function sample_player_2_action(pi2::Matrix{Float64}, s_idx::Int)::Int
    return rand(Categorical(pi2[s_idx, :]))
end


function calculate_value(alpha_vectors::Vector{<:Vector{Float64}}, belief::Vector{Float64})::Float64    
    if isempty(alpha_vectors)        
        return -Inf
    end    
    if length(belief) != length(alpha_vectors[1])
        throw(DimensionMismatch(
            "Belief vector (length $(length(belief))) and alpha vectors (length $(length(alpha_vectors[1]))) must have the same size."
        ))
    end    
    dot_products = (dot(belief, alpha) for alpha in alpha_vectors)
    return maximum(dot_products)
end