using Pkg
Pkg.activate(".")
using SAB

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

B_n = AggregationUtil.generate_aggregate_belief_space(n, length(S))
println("Num representative beliefs: $(size(B_n, 1))")
PI2 = AggregationUtil.generate_aggregate_action_space_of_player_2(m, length(A2), length(S))
T_b = AggregationUtil.generate_aggregate_belief_transition_operator_sparse_ann(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
R_b = AggregationUtil.generate_aggregate_belief_reward_tensor(B_n, PI2, R)

V, maximin_strategies, minimax_strategies, auxillary_games, deltas = ShapleyIteration.shapley_iteration_sparse(
    B_n, max_iterations, gamma, length(A1), size(PI2, 1), R_b, T_b; 
    delta_threshold=delta_threshold, log_every=1)

nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(B_n, b1)
println("Value: $(V[nearest_belief_index])")

for b1 in 0:0.01:1
    b0 = 1-b1
    b = [b0, b1]
    nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(B_n, b)
    value = V[nearest_belief_index]
    println("$b1 $value")
end