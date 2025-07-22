using Pkg
Pkg.activate(".")
using SAB


gamma = 0.99
N = 4
p_detect = 1.0
t_A = 3
p = 0.2

n = 5
m = 1
delta_threshold = 0.1
max_iterations = 1000

graph = PatrollingGame.generate_graph(N, p)
b1 = PatrollingGame.b1(N, t_A)
A1 = PatrollingGame.player_1_actions(N)
A2 = PatrollingGame.player_2_actions(graph)
S = PatrollingGame.state_space(N, t_A)
O = PatrollingGame.observation_space()
Z = PatrollingGame.observation_tensor(N, t_A, graph, p_detect)
T = PatrollingGame.transition_tensor(N, t_A, graph)
R = PatrollingGame.reward_tensor(N, t_A, graph)

OsPosgFile.generate_os_posg_game_file(length(S), length(A1), length(A2), length(O), Z, T, R, gamma, b1, "patrolling_4.posg")