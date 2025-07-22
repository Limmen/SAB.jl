using Pkg
Pkg.activate(".")
using SAB
using LinearAlgebra

num_obs = 10
gamma = 0.99
N = 1
p_a = 0.2

n = 20
m = 5
delta_threshold = 0.01
max_iterations = 1000

b1 = AptGame.b1(N)
A1 = AptGame.player_1_actions()
A2 = AptGame.player_2_actions()
S = AptGame.state_space(N)
O = AptGame.observation_space(num_obs)
Z = AptGame.observation_tensor(num_obs, N)
T = AptGame.transition_tensor(N, p_a)
R = AptGame.reward_tensor(N)

file_path = "V.csv"
alpha_vectors = OsPosgFile.parse_value_function(file_path)
upper_bounds = []
for n in 1:100
    B_n = AggregationUtil.generate_aggregate_belief_space(n, length(S))
    PI2 = AggregationUtil.generate_aggregate_action_space_of_player_2(m, length(A2), length(S))
    T_b = AggregationUtil.generate_aggregate_belief_transition_operator_sparse_ann(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
    R_b = AggregationUtil.generate_aggregate_belief_reward_tensor(B_n, PI2, R)
    V, maximin_strategies, minimax_strategies, auxillary_games, deltas = ShapleyIteration.shapley_iteration_sparse(
        B_n, max_iterations, gamma, length(A1), size(PI2, 1), R_b, T_b;
        delta_threshold=delta_threshold, log_every=1, verbose=false)
    V_tilde = []
    V_star = []    
    footprint_sets = [Vector{Int}() for _ in 1:size(B_n, 1)]    

    for b1 in 0:0.01:1
        b0 = 1 - b1
        b = [b0, b1]        
        nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(B_n, b)
        push!(footprint_sets[nearest_belief_index], nearest_belief_index)        
        value = V[nearest_belief_index]
        optimal_value = OsPosgUtil.calculate_value(alpha_vectors, b)
        push!(V_tilde, value)
        push!(V_star, optimal_value)
    end
    footprint_sets_values = []
    for footprint_set_index in 1:size(B_n, 1)
        footprint_set = footprint_sets[footprint_set_index]
        V_tilde_footprint = []
        V_star_footprint = []
        for i in 1:length(footprint_set)
            #nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(B_n, footprint_set[i]
            b_idx = footprint_set[i]
            value = V[b_idx]
            optimal_value = OsPosgUtil.calculate_value(alpha_vectors, B_n[b_idx, :])
            push!(V_tilde_footprint, value)
            push!(V_star_footprint, optimal_value)
        end
        push!(footprint_sets_values, norm(V_tilde_footprint-V_star_footprint, Inf))        
    end
    epsilon = maximum(footprint_sets_values)
    upper_bound = epsilon/(1-gamma)
    push!(upper_bounds, upper_bound)
    ub = minimum(upper_bounds)    
    #println("$(n) $(norm(V_tilde - V_star, Inf)) $(ub)")
    println("$(n) $(ub)")
end
