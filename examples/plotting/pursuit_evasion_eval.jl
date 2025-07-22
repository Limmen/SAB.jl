using Pkg
Pkg.activate(".")
using SAB
using LinearAlgebra

gamma = 0.99
N = 4
p_c = 0.5

n = 3
m = 1
delta_threshold = 0.1
max_iterations = 2000
threshold = 0.15


b1 = PursuitEvasionGame.b1(N)
A1 = PursuitEvasionGame.player_1_actions(N)
A2 = PursuitEvasionGame.player_2_actions()
S = PursuitEvasionGame.state_space(N)
O = PursuitEvasionGame.observation_space()
Z = PursuitEvasionGame.observation_tensor(N, p_c)
T = PursuitEvasionGame.transition_tensor(N, p_c)
R = PursuitEvasionGame.reward_tensor(N)

# 165 beliefs


println("Number of states: $(length(S))")
println("Number of actions of player 1: $(length(A1))")
println("Number of actions of player 2: $(length(A2))")

B_eval = AggregationUtil.generate_aggregate_belief_space(4, length(S))
println("B_eval dimension = $(size(B_eval, 1))")

elapsed_time = @elapsed B_n = AggregationUtil.generate_aggregate_belief_space(n, length(S))
println("B_n creation: $(round(elapsed_time, digits=3)) seconds")
println("Num representative beliefs: $(size(B_n, 1))")

file_path = "V_peg_$(N).csv"
alpha_vectors = OsPosgFile.parse_value_function(file_path)

V_star = Array{Float64, 1}()
for i in 1:size(B_eval, 1)
    b = B_eval[i, :]     
    optimal_value = OsPosgUtil.calculate_value(alpha_vectors, b)    
    push!(V_star, optimal_value)
end

elapsed_time = @elapsed V = ShapleyIteration.shapley_iteration_dynamic(
    B_n, length(A2), R, T, Z, max_iterations, gamma, V_star, B_eval, threshold; delta_threshold=delta_threshold, log_every=1)

println("Shapley iteration: $(round(elapsed_time, digits=3)) seconds")

B_eval = AggregationUtil.generate_aggregate_belief_space(5, length(S))
V_tilde = []
V_star = []
for i in 1:size(B_eval, 1)
    b = B_eval[i, :] 
    nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(B_n, b)
    value = V[nearest_belief_index]
    optimal_value = OsPosgUtil.calculate_value(alpha_vectors, b)    
    push!(V_star, optimal_value)
    push!(V_tilde, value)    
end

println("$(norm(V_tilde - V_star, Inf))")