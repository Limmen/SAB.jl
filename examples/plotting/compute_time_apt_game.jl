using Pkg
Pkg.activate(".")
using SAB
using LinearAlgebra

num_obs = 10
gamma = 0.99
N = 1
p_a = 0.2

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
    elapsed_time_1 = @elapsed T_b = AggregationUtil.generate_aggregate_belief_transition_operator_sparse_ann(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
    R_b = AggregationUtil.generate_aggregate_belief_reward_tensor(B_n, PI2, R)
    elapsed_time_2 = @elapsed V, maximin_strategies, minimax_strategies, auxillary_games, deltas = ShapleyIteration.shapley_iteration_sparse(
        B_n, max_iterations, gamma, length(A1), size(PI2, 1), R_b, T_b;
        delta_threshold=delta_threshold, log_every=1, verbose=false)    
    println("$(n) $(elapsed_time_1 + elapsed_time_2)")
end
