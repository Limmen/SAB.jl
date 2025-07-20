using LinearAlgebra


function b1(N::Integer)::Vector{Float64}
    num_states = 3 * N
    prob = 1.0 / num_states
    return fill(prob, num_states)
end


function state_to_idx(state::Tuple{Int,Int}, N::Int)::Int
    row, col = state
    return (row - 1) * N + col
end


function idx_to_state(idx::Int, N::Int)::Tuple{Int,Int}
    row = div(idx - 1, N) + 1
    col = (idx - 1) % N + 1
    return (row, col)
end

function state_space(N::Int)::Vector{Tuple{Int,Int}}
    states = []
    for r in 1:3, c in 1:N
        push!(states, (r, c))
    end
    return states
end

observation_space() = collect(0:1)

function player_1_actions(N::Int)::Vector{Tuple{Int,Int}}
    return state_space(N) # Actions are the grid cells
end

function player_2_actions()::Vector{Tuple{Int,Int}}
    return [
        (0, 0),  # Action 1: stay
        (-1, 0), # Action 2: up
        (1, 0),  # Action 3: down
        (0, -1), # Action 4: left
        (0, 1)   # Action 5: right
    ]
end

function reward_tensor(N::Int)::Array{Float64,3}
    num_a1 = 3 * N
    num_a2 = length(player_2_actions())
    num_s = 3 * N
    R = zeros(num_a1, num_a2, num_s)

    for s_idx in 1:num_s
        evader_pos = idx_to_state(s_idx, N)
        for a1_idx in 1:num_a1
            pursuer_search_pos = idx_to_state(a1_idx, N)
            if pursuer_search_pos == evader_pos
                # The reward is independent of the evader's action
                R[a1_idx, :, s_idx] .= 1.0
            end
        end
    end
    return R
end

function transition_tensor(N::Int, p_c::Float64)::Array{Float64,4}
    p1_actions = player_1_actions(N)
    p2_actions = player_2_actions() # Directly get the vector of actions

    num_a1 = length(p1_actions)
    num_a2 = length(p2_actions)
    num_s = 3 * N

    T = zeros(num_a1, num_a2, num_s, num_s)

    uniform_prob = 1.0 / num_s

    for s_idx in 1:num_s
        current_pos = idx_to_state(s_idx, N)

        for a1_idx in 1:num_a1
            search_pos = idx_to_state(a1_idx, N)

            for a2_idx in 1:num_a2
                move = p2_actions[a2_idx]
                next_pos_row = current_pos[1] + move[1]
                next_pos_col = current_pos[2] + move[2]

                # Check for valid move (within grid boundaries)
                if 1 <= next_pos_row <= 3 && 1 <= next_pos_col <= N
                    next_s_idx = state_to_idx((next_pos_row, next_pos_col), N)

                    if current_pos == search_pos # Pursuer searches the correct cell
                        # With probability p_c, evader is caught and reset
                        for reset_s_idx in 1:num_s
                            T[a1_idx, a2_idx, s_idx, reset_s_idx] += p_c * uniform_prob
                        end
                        # With probability 1-p_c, evader escapes to the next cell
                        T[a1_idx, a2_idx, s_idx, next_s_idx] += (1 - p_c)
                    else # Pursuer searches the wrong cell
                        # Evader moves to the adjacent cell with certainty
                        T[a1_idx, a2_idx, s_idx, next_s_idx] = 1.0
                    end
                else
                    # If the move is invalid, the evader stays in the current position
                    if current_pos == search_pos
                        # With probability p_c, evader is caught and reset
                        for reset_s_idx in 1:num_s
                            T[a1_idx, a2_idx, s_idx, reset_s_idx] += p_c * uniform_prob
                        end
                        # With probability 1-p_c, evader fails to move and stays
                        T[a1_idx, a2_idx, s_idx, s_idx] += (1 - p_c)
                    else
                        # Evader fails to move and stays
                        T[a1_idx, a2_idx, s_idx, s_idx] = 1.0
                    end
                end
            end
        end
    end
    return T
end

function observation_tensor(N::Int, p_c::Float64)::Array{Float64, 4}
    num_a1 = 3 * N
    num_a2 = length(player_2_actions())
    num_s = 3 * N
    num_obs = 2 # {1: NOT_FOUND, 2: FOUND}
    
    Z = zeros(num_a1, num_a2, num_s, num_obs)

    for s_prime_idx in 1:num_s
        evader_pos = idx_to_state(s_prime_idx, N)
        for a1_idx in 1:num_a1
            search_pos = idx_to_state(a1_idx, N)
            
            # The observation model is independent of player 2's action (a2).
            # We use the broadcast operator `.=` to assign a value to a slice.
            if search_pos == evader_pos # Pursuer searched the correct cell.
                Z[a1_idx, :, s_prime_idx, 1] .= 1.0 - p_c   # P(NOT_FOUND)
                Z[a1_idx, :, s_prime_idx, 2] .= p_c         # P(FOUND)
            else # Pursuer searched the wrong cell.
                Z[a1_idx, :, s_prime_idx, 1] .= 1.0  # P(NOT_FOUND)
                Z[a1_idx, :, s_prime_idx, 2] .= 0.0  # P(FOUND)
            end
        end
    end
    return Z
end