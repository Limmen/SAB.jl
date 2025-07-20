using Graphs
using LinearAlgebra

observation_space() = collect(0:1)

function generate_graph(N::Integer, p::Float64)::SimpleGraph
    return erdos_renyi(N, p)
end

function state_to_idx(s::Tuple{Int,Int}, t_A::Integer)::Int
    location, time = s
    return (location - 1) * t_A + time
end

function idx_to_state(idx::Integer, N::Integer, t_A::Integer)::Tuple{Int,Int}
    # This prevents errors for idx=0 or invalid inputs
    @assert idx > 0
    location = div(idx - 1, t_A) + 1
    time = (idx - 1) % t_A + 1
    return (location, time)
end

function b1(N::Integer, t_A::Integer)::Vector{Float64}
    num_states = N * t_A
    b = zeros(num_states)
    prob_per_node = 1.0 / N
    for u in 1:N
        s_idx = state_to_idx((u, 1), t_A)
        b[s_idx] = prob_per_node
    end
    return b
end

function state_space(N::Integer, t_A::Integer)::Vector{Tuple{Int,Int}}
    states = []
    for n in 1:N, t in 1:t_A
        push!(states, (n, t))
    end
    return states
end

function player_1_actions(N::Integer)::Vector{Int}
    return collect(1:N)
end


function player_2_actions(graph::SimpleGraph)::Vector{Int}
    d_max = isempty(vertices(graph)) ? 0 : maximum(degree(graph))
    return collect(0:d_max)
end

function reward_tensor(N::Integer, t_A::Integer, graph::SimpleGraph)::Array{Float64,3}
    num_a1 = length(player_1_actions(N))
    num_a2 = length(player_2_actions(graph))
    num_s = N * t_A
    R = zeros(num_a1, num_a2, num_s)

    for s_idx in 1:num_s
        loc, time = idx_to_state(s_idx, N, t_A)
        for a1 in 1:num_a1
            if a1 == loc  # Patroller visits the attacker's location
                R[a1, :, s_idx] .= 1.0 # Detection reward, independent of attacker action
            elseif time == t_A # Attack successful (patroller is elsewhere)
                R[a1, :, s_idx] .= -1.0
            end
        end
    end
    return R
end

function transition_tensor(N::Integer, t_A::Integer, graph::SimpleGraph)::Array{Float64,4}
    p1_actions = player_1_actions(N)
    p2_actions = player_2_actions(graph)
    num_a1 = length(p1_actions)
    num_a2 = length(p2_actions)
    num_s = N * t_A
    T = zeros(num_a1, num_a2, num_s, num_s)

    uniform_reset_prob = 1.0 / N

    for s_idx in 1:num_s
        loc, time = idx_to_state(s_idx, N, t_A)

        for a1 in 1:num_a1
            # Case 1: Detection or Successful Attack -> Reset
            if a1 == loc || time == t_A
                for next_loc in 1:N
                    s_prime_idx = state_to_idx((next_loc, 1), t_A)
                    T[a1, :, s_idx, s_prime_idx] .= uniform_reset_prob
                end
                continue # End of logic for this state-action
            end

            # Case 2: No detection, no successful attack -> Attacker moves
            neighbors_of_loc = neighbors(graph, loc)
            for a2_idx in 1:num_a2
                a2 = p2_actions[a2_idx]

                if a2 == 0 # Attacker stays
                    s_prime = (loc, time + 1)
                    s_prime_idx = state_to_idx(s_prime, t_A)
                    T[a1, a2_idx, s_idx, s_prime_idx] = 1.0
                elseif a2 <= length(neighbors_of_loc) # Attacker moves to a valid neighbor
                    next_loc = neighbors_of_loc[a2]
                    s_prime = (next_loc, 1) # Time resets on move
                    s_prime_idx = state_to_idx(s_prime, t_A)
                    T[a1, a2_idx, s_idx, s_prime_idx] = 1.0
                else # Attacker chose an invalid move (e.g., action 3 for a node with 2 neighbors)
                    # Assumption: invalid move results in staying put
                    s_prime = (loc, time + 1)
                    s_prime_idx = state_to_idx(s_prime, t_A)
                    T[a1, a2_idx, s_idx, s_prime_idx] = 1.0
                end
            end
        end
    end
    return T
end

function observation_tensor(N::Integer, t_A::Integer, graph::SimpleGraph, p_detect::Float64)::Array{Float64,4}
    num_a1 = length(player_1_actions(N))
    num_a2 = length(player_2_actions(graph))
    num_s = N * t_A
    num_obs = length(observation_space())

    Z = zeros(num_a1, num_a2, num_s, num_obs)

    for s_prime_idx in 1:num_s
        loc_prime, _ = idx_to_state(s_prime_idx, N, t_A)
        for a1 in 1:num_a1
            # Observation is independent of player 2's action (a2)
            if a1 == loc_prime # Patroller visits the attacker's new location
                Z[a1, :, s_prime_idx, 1] .= 1.0 - p_detect
                Z[a1, :, s_prime_idx, 2] .= p_detect
            else # Patroller visits any other location
                Z[a1, :, s_prime_idx, 1] .= 1.0
                Z[a1, :, s_prime_idx, 2] .= 0.0
            end
        end
    end
    return Z
end