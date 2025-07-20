using Distributions


b1() = [1.0, 0.0, 0.0]
state_space() = collect(0:2)
player_1_actions() = collect(0:1)
player_2_actions() = collect(0:1)
observation_space(n::Int) = collect(0:n)

function reward_tensor()::Array{Float64, 3}
    num_a1 = length(player_1_actions())
    num_a2 = length(player_2_actions())
    num_s = length(state_space())
    R = zeros(num_a1, num_a2, num_s)
    # --- Fill in the values based on the game logic ---
    # a1 = 1 (Defender continues)
    # a2 = 1 (Attacker continues)
    R[1, 1, 2] = -1.0
    # a2 = 2 (Attacker stops) -> rewards are 0
    # a1 = 2 (Defender stops)
    # a2 = 1 (Attacker continues)
    R[2, 1, 1] = -2.0
    R[2, 1, 2] = 2.0
    # a2 = 2 (Attacker stops)
    R[2, 2, 1] = -2.0
    return R
end

function transition_tensor()::Array{Float64, 4}
    # T[a1, a2, s, s']
    T = zeros(2, 2, 3, 3)

    # Defender continues (a1=1), Attacker continues (a2=1) -> Identity
    T[1, 1, :, :] = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

    # Defender continues (a1=1), Attacker stops (a2=2)
    T[1, 2, 1, 2] = 1.0 # No intrusion -> Intrusion
    T[1, 2, 2, 3] = 1.0 # Intrusion -> Terminal
    T[1, 2, 3, 3] = 1.0 # Terminal -> Terminal

    # Defender stops (a1=2) -> Always go to terminal state
    T[2, :, :, 3] .= 1.0

    return T
end


function observation_tensor(n::Int)::Array{Float64, 4}    
    intrusion_dist_generator = BetaBinomial(n, 1.0, 0.7)
    no_intrusion_dist_generator = BetaBinomial(n, 0.7, 3.0)
    
    obs_range = 0:n
    intrusion_dist = [pdf(intrusion_dist_generator, i) for i in obs_range]
    no_intrusion_dist = [pdf(no_intrusion_dist_generator, i) for i in obs_range]

    terminal_dist = zeros(n + 1)
    terminal_dist[end] = 1.0
    
    Z = zeros(2, 2, 3, n + 1)
    for a1 in 1:2, a2 in 1:2
        Z[a1, a2, 1, :] = no_intrusion_dist
        Z[a1, a2, 2, :] = intrusion_dist
        Z[a1, a2, 3, :] = terminal_dist
    end

    return Z
end