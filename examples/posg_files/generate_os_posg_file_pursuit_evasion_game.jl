using Pkg
Pkg.activate(".")
using SAB

gamma = 0.99
N = 1
p_c = 0.5

n = 5
m = 1
delta_threshold = 0.1
max_iterations = 1000

b1 = PursuitEvasionGame.b1(N)
A1 = PursuitEvasionGame.player_1_actions(N)
A2 = PursuitEvasionGame.player_2_actions()
S = PursuitEvasionGame.state_space(N)
O = PursuitEvasionGame.observation_space()
Z = PursuitEvasionGame.observation_tensor(N, p_c)
T = PursuitEvasionGame.transition_tensor(N, p_c)
R = PursuitEvasionGame.reward_tensor(N)

OsPosgFile.generate_os_posg_game_file(length(S), length(A1), length(A2), length(O), Z, T, R, gamma, b1, "pursuit_evasion_1.posg")