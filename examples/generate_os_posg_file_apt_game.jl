using Pkg
Pkg.activate(".")
using SAB


num_obs = 10
gamma = 0.99
N = 2
p_a = 0.2

b1 = AptGame.b1(N)
A1 = AptGame.player_1_actions()
A2 = AptGame.player_2_actions()
S = AptGame.state_space(N)
O = AptGame.observation_space(num_obs)
Z = AptGame.observation_tensor(num_obs, N)
T = AptGame.transition_tensor(N, p_a)
R = AptGame.reward_tensor(N)

OsPosgFile.generate_os_posg_game_file(length(S), length(A1), length(A2), length(O), Z, T, R, gamma, b1, "apt_game.posg")