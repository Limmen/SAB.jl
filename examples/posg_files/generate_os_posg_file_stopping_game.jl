using Pkg
Pkg.activate(".")
using SAB

num_obs = 10
gamma = 0.99

b1 = StoppingGame.b1()
A1 = StoppingGame.player_1_actions()
A2 = StoppingGame.player_2_actions()
S = StoppingGame.state_space()
O = StoppingGame.observation_space(num_obs)
Z = StoppingGame.observation_tensor(num_obs)
T = StoppingGame.transition_tensor()
R = StoppingGame.reward_tensor()

OsPosgFile.generate_os_posg_game_file(length(S), length(A1), length(A2), length(O), Z, T, R, gamma, b1, "stopping_game.posg")