using Pkg
Pkg.activate(".")
using SAB
using LinearAlgebra

gamma = 0.99
N = 4
p_detect = 1.0
t_A = 3
p = 0.2

n = 4
m = 1
delta_threshold = 0.1
max_iterations = 2000
threshold = 0.0


graph = PatrollingGame.generate_graph(N, p)
b1 = PatrollingGame.b1(N, t_A)
A1 = PatrollingGame.player_1_actions(N)
A2 = PatrollingGame.player_2_actions(graph)
S = PatrollingGame.state_space(N, t_A)
O = PatrollingGame.observation_space()
Z = PatrollingGame.observation_tensor(N, t_A, graph, p_detect)
T = PatrollingGame.transition_tensor(N, t_A, graph)
R = PatrollingGame.reward_tensor(N, t_A, graph)

println("Number of states: $(length(S))")
println("Number of actions of player 1: $(length(A1))")
println("Number of actions of player 2: $(length(A2))")

B_eval = AggregationUtil.generate_aggregate_belief_space(4, length(S))
println("B_eval dimension = $(size(B_eval, 1))")

elapsed_time = @elapsed B_n = AggregationUtil.generate_aggregate_belief_space(n, length(S))
println("B_n creation: $(round(elapsed_time, digits=3)) seconds")
println("Num representative beliefs: $(size(B_n, 1))")

file_path = "V_patrolling_$(N).csv"
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