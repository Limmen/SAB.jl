@testset "stopping_game_functions tests" begin
    @test StoppingGame.b1() == [1.0, 0.0, 0.0]
    @test StoppingGame.state_space() == [0, 1, 2]
    @test StoppingGame.player_1_actions() == [0, 1]
    @test StoppingGame.player_2_actions() == [0, 1]
    @test StoppingGame.observation_space(5) == [0, 1, 2, 3 ,4, 5]
    @test size(StoppingGame.reward_tensor()) == (length(StoppingGame.player_1_actions()), length(StoppingGame.player_2_actions()), length(StoppingGame.state_space()))
    @test size(StoppingGame.transition_tensor()) == (length(StoppingGame.player_1_actions()), length(StoppingGame.player_2_actions()), length(StoppingGame.state_space()), length(StoppingGame.state_space()))
    @test size(StoppingGame.observation_tensor(5)) == (length(StoppingGame.player_1_actions()), length(StoppingGame.player_2_actions()), length(StoppingGame.state_space()), 6)
end