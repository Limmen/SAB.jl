

    function b1(N::Integer)::Array{Float64,1}
        b1 = zeros(N+1)
        b1[1] = 1.0
        return b1
    end

    state_space(N) = collect(0:N)
    player_1_actions() = collect(0:1)
    player_2_actions() = collect(0:1)
    observation_space(n::Int) = collect(0:n)

    function cost_function(s_idx::Integer, a1_idx::Integer)::Float64
        s = s_idx - 1
        a1 = a1_idx - 1
        return s^(5/4)*(1-a1) + a1-2*(a1)*sign(s)
    end

    function reward_tensor(N::Integer)::Array{Float64, 3}
        num_a1 = length(player_1_actions())
        num_a2 = length(player_2_actions())
        num_s = length(state_space(N))
        R = zeros(num_a1, num_a2, num_s)
        for s in 1:num_s, a1 in 1:num_a1, a2 in 1:num_a2
            R[a1, a2, s] = -cost_function(s, a1)
        end    
        return R
    end

    function transition_tensor(N::Integer, p_a::Float64)::Array{Float64, 4}
        num_a1 = length(player_1_actions())
        num_a2 = length(player_2_actions())
        num_s = length(state_space(N))
        T = zeros(num_a1, num_a2, num_s, num_s)
        for a1 in 1:num_a1, a2 in 1:num_a2
            for s in 1:num_s, s_prime in 1:num_s
                if a1 == 2 && s_prime == 1
                    T[a1, a2, s, s_prime] = 1.0
                elseif a1 == 1 && a2 == 1 && s_prime == s
                    T[a1, a2, s, s_prime] = 1.0
                elseif a1 == 1 && s == N+1 && s_prime == N+1
                    T[a1, a2, s, s_prime] = 1.0
                elseif a1 == 1 && a2 == 2 && s_prime == s
                    T[a1, a2, s, s_prime] = 1.0-p_a
                elseif a1 == 1 && a2 == 2 && s_prime == s + 1
                    T[a1, a2, s, s_prime] = p_a
                end
            end
        end
        return T
    end

    function observation_tensor(n::Int, N::Integer)::Array{Float64, 4}    
        intrusion_dist_generator = BetaBinomial(n, 1.0, 0.7)
        no_intrusion_dist_generator = BetaBinomial(n, 0.7, 3.0)
        
        obs_range = 0:n        
        Z = zeros(2, 2, N+1, n+1)
        for s_prime in 1:N+1, a1 in 1:2, a2 in 1:2            
            if s_prime == 0
                Z[a1, a2, s_prime, :] = [pdf(no_intrusion_dist_generator, i) for i in obs_range]
            else
                Z[a1, a2, s_prime, :] = [pdf(intrusion_dist_generator, i) for i in obs_range]
            end            
        end
        return Z
    end