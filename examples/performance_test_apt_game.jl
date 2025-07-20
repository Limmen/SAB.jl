using Pkg
Pkg.activate(".")
using SAB

num_obs = 10
gamma = 0.99
N = 2
p_a = 0.2

n = 20
m = 2
delta_threshold = 0.1
max_iterations = 1000

b1 = AptGame.b1(N)
A1 = AptGame.player_1_actions()
A2 = AptGame.player_2_actions()
S = AptGame.state_space(N)
O = AptGame.observation_space(num_obs)
Z = AptGame.observation_tensor(num_obs, N)
T = AptGame.transition_tensor(N, p_a)
R = AptGame.reward_tensor(N)

println("T dimension = $(size(T))")
println("Z dimension = $(size(Z))")
println("R dimension = $(size(R))")

elapsed_time = @elapsed B_n = AggregationUtil.generate_aggregate_belief_space(n, length(S))
println("B_n creation: $(round(elapsed_time, digits=3)) seconds")
println("Num representative beliefs: $(size(B_n, 1))")
elapsed_time = @elapsed PI2 = AggregationUtil.generate_aggregate_action_space_of_player_2(m, length(A2), length(S))
println("PI2 creation: $(round(elapsed_time, digits=3)) seconds")
println("Num PI2: $(size(PI2, 1))")
#elapsed_time = @elapsed T_b = AggregationUtil.generate_aggregate_belief_transition_operator(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
#println("T_b creation: $(round(elapsed_time, digits=3)) seconds")
#elapsed_time = @elapsed T_b = AggregationUtil.generate_aggregate_belief_transition_operator_ann(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
elapsed_time = @elapsed T_b = AggregationUtil.generate_aggregate_belief_transition_operator_sparse_ann(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
println("T_b_approx creation: $(round(elapsed_time, digits=3)) seconds")
elapsed_time = @elapsed R_b = AggregationUtil.generate_aggregate_belief_reward_tensor(B_n, PI2, R)
println("R_b creation: $(round(elapsed_time, digits=3)) seconds")

println("B_n dimension = $(size(B_n))")
println("T_b dimension = $(size(T_b))")
println("R_b dimension = $(size(R_b))")

#elapsed_time = @elapsed V, maximin_strategies, minimax_strategies, auxillary_games, deltas = ShapleyIteration.shapley_iteration(
#    B_n, max_iterations, gamma, length(A1), size(PI2, 1), R_b, T_b; 
#    delta_threshold=delta_threshold, log_every=1)

elapsed_time = @elapsed V, maximin_strategies, minimax_strategies, auxillary_games, deltas = ShapleyIteration.shapley_iteration_sparse(
    B_n, max_iterations, gamma, length(A1), size(PI2, 1), R_b, T_b; 
    delta_threshold=delta_threshold, log_every=1)

println("Shapley iteration: $(round(elapsed_time, digits=3)) seconds")
nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(B_n, b1)
println("Value: $(V[nearest_belief_index])")
