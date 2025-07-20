module PatrollingGame

using Distributions, Graphs, LinearAlgebra

include("patrolling_game_functions.jl")

export b1, player_1_actions, player_2_actions, state_space, observation_space, observation_tensor, transition_tensor, reward_tensor

end