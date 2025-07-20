module StoppingGame

using Distributions

include("stopping_game_functions.jl")

export b1, state_space, player_1_actions, player_2_actions, observation_space,
       reward_tensor, transition_tensor, observation_tensor

end