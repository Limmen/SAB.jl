using Pkg
Pkg.activate(".")
using SAB

gamma = 0.99
N = 1
p_c = 0.5

n = 2
m = 1
delta_threshold = 0.01
max_iterations = 1000

b1 = PursuitEvasionGame.b1(N)
A1 = PursuitEvasionGame.player_1_actions(N)
A2 = PursuitEvasionGame.player_2_actions()
S = PursuitEvasionGame.state_space(N)
O = PursuitEvasionGame.observation_space()
Z = PursuitEvasionGame.observation_tensor(N, p_c)
T = PursuitEvasionGame.transition_tensor(N, p_c)
R = PursuitEvasionGame.reward_tensor(N)

println("Number of states: $(length(S))")
println("Number of actions of player 1: $(length(A1))")
println("Number of actions of player 2: $(length(A2))")

elapsed_time = @elapsed B_n = AggregationUtil.generate_aggregate_belief_space(n, length(S))
println("B_n creation: $(round(elapsed_time, digits=3)) seconds")
println("Num representative beliefs: $(size(B_n, 1))")

elapsed_time = @elapsed V = ShapleyIteration.shapley_iteration_dynamic(
    B_n, length(A2), R, T, Z, max_iterations, gamma; delta_threshold=delta_threshold, log_every=1)

println("Shapley iteration: $(round(elapsed_time, digits=3)) seconds")
nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(B_n, b1)
println("Value: $(V[nearest_belief_index])")
