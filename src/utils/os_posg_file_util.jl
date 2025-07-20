function generate_transitions(num_states::Int, num_a1::Int, num_a2::Int, num_obs::Int, Z::Array{Float64,4}, T::Array{Float64,4})::Vector{String}
    transitions = String[]    
    for s in 1:num_states
        for a1 in 1:num_a1
            for a2 in 1:num_a2
                for s_prime in 1:num_states
                    for o in 1:num_obs
                        tr_prob = T[a1, a2, s, s_prime]
                        obs_prob = Z[a1, a2, s_prime, o]
                        prob = tr_prob * obs_prob
                        if prob > 0                            
                            push!(transitions, "$(s-1) $(a1-1) $(a2-1) $(o-1) $(s_prime-1) $prob")
                        end
                    end
                end
            end
        end
    end
    return transitions
end

function generate_rewards(num_states::Int, num_a1::Int, num_a2::Int, R::Array{Float64,3})::Vector{String}
    rewards = String[]
    for s in 1:num_states
        for a1 in 1:num_a1
            for a2 in 1:num_a2
                reward = R[a1, a2, s]
                if reward != 0.0
                    push!(rewards, "$(s-1) $(a1-1) $(a2-1) $reward")
                end
            end
        end
    end
    return rewards
end

function generate_os_posg_game_file(num_states::Int, num_a1::Int, num_a2::Int, num_obs::Int, Z::Array{Float64,4},
    T::Array{Float64,4}, R::Array{Float64,3}, gamma::Float64, b1::Vector{Float64}, file_name::String)
    num_partitions = 1
    transitions = generate_transitions(num_states, num_a1, num_a2, num_obs, Z, T)
    rewards = generate_rewards(num_states, num_a1, num_a2, R)
    game_description = "$num_states $num_partitions $num_a1 $num_a2 $num_obs $(length(transitions)) $(length(rewards)) $gamma"
    state_descriptions = ["$(s-1) 0" for s in 1:num_states]
    player_1_actions = ["a1_$(i-1)" for i in 1:num_a1]
    player_2_actions = ["a2_$(i-1)" for i in 1:num_a2]
    obs_desriptions = ["o_$(o-1)" for o in 1:num_obs]
    player_2_legal_actions = [join(0:(num_a2-1), " ") for _ in 1:num_states]
    player_1_legal_actions = [join(0:(num_a1-1), " ")]
    initial_belief_str = "0 $(join(b1, " "))"
    file_parts = [
        game_description,
        join(state_descriptions, "\n"),
        join(player_1_actions, "\n"),
        join(player_2_actions, "\n"),
        join(obs_desriptions, "\n"),
        join(player_2_legal_actions, "\n"),
        join(player_1_legal_actions, "\n"),
        join(transitions, "\n"),
        join(rewards, "\n"),
        initial_belief_str
    ]

    open(file_name, "w") do f
        write(f, join(file_parts, "\n"))
    end
end


function parse_value_function(file_path::String)::Vector{Vector{Float64}}    
    if !isfile(file_path)
        error("File not found at path: $file_path")
    end    
    df = CSV.read(file_path, DataFrame, skipto=2, header=false)    
    alpha_vectors = Vector{Vector{Float64}}()    
    for row in eachrow(df)        
        alpha_vector = Vector(row)[2:end]
        push!(alpha_vectors, alpha_vector)
    end
    return alpha_vectors
end