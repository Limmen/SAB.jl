@testset "apt_game_functions tests" begin
    @test AptGame.b1(1) == [1.0, 0.0]
    @test AptGame.b1(2) == [1.0, 0.0, 0.0]
    @test AptGame.state_space(1) == [0, 1]
    @test AptGame.state_space(2) == [0, 1, 2]
    @test AptGame.player_1_actions() == [0, 1]
    @test AptGame.player_2_actions() == [0, 1]
    @test AptGame.observation_space(5) == [0, 1, 2, 3 ,4, 5]
    @test AptGame.cost_function(1, 2) == 1.0
    @test AptGame.cost_function(2, 2) == -1.0
    @test AptGame.cost_function(2, 1) == 1.0
    N=3
    p_a = 0.2
    @test size(AptGame.reward_tensor(N)) == (length(AptGame.player_1_actions()), length(AptGame.player_2_actions()), length(AptGame.state_space(N)))
    @test size(AptGame.transition_tensor(N, p_a)) == (length(AptGame.player_1_actions()), length(AptGame.player_2_actions()), length(AptGame.state_space(N)), length(AptGame.state_space(N)))
    @test size(AptGame.observation_tensor(5, N)) == (length(AptGame.player_1_actions()), length(AptGame.player_2_actions()), length(AptGame.state_space(N)), 6)

    T = AptGame.transition_tensor(N, p_a)
    @test T[1, 1, 1, 1] == 1.0
    @test T[1, 1, 2, 2] == 1.0
    @test T[1, 1, 3, 3] == 1.0
    @test T[1, 1, 4, 4] == 1.0
    
    @test T[2, 1, 1, 1] == 1.0
    @test T[2, 1, 2, 1] == 1.0
    @test T[2, 1, 3, 1] == 1.0
    @test T[2, 1, 4, 1] == 1.0
    
    @test T[2, 2, 1, 1] == 1.0
    @test T[2, 2, 2, 1] == 1.0
    @test T[2, 2, 3, 1] == 1.0
    @test T[2, 2, 4, 1] == 1.0

    @test T[1, 2, 4, 4] == 1.0

    @test T[1, 2, 1, 2] == p_a
    @test T[1, 2, 2, 3] == p_a
    @test T[1, 2, 3, 4] == p_a

    @test T[1, 2, 1, 1] == 1-p_a
    @test T[1, 2, 2, 2] == 1-p_a
    @test T[1, 2, 3, 3] == 1-p_a

    R = AptGame.reward_tensor(N)
    @test R[1,1,2] == -1.0
    @test R[2,1,2] == 1.0
    @test R[2,1,1] == -1.0
end