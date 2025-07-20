
@testset "aggregation_util Tests" begin

    @testset "Combinatorial Helpers" begin        
        compositions_result = AggregationUtil._integer_compositions(2, 2)
        @test Set(compositions_result) == Set([[2, 0], [1, 1], [0, 2]])
        @test all(sum(c) == 2 for c in compositions_result)
        @test AggregationUtil._integer_compositions(0, 3) == [[0, 0, 0]]
                
        belief_space = AggregationUtil.generate_aggregate_belief_space(2, 2)
        @test belief_space isa Matrix{Float64}
        @test size(belief_space) == (3, 2)
        @test all(sum(row) ≈ 1.0 for row in eachrow(belief_space))        
        @test Set(eachrow(belief_space)) == Set([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    end

    @testset "Action Space Generation" begin
        n, S, A2 = 1, collect(1:2), collect(1:2)
        PI2 = AggregationUtil.generate_aggregate_action_space_of_player_2(n, length(A2), length(S))
        @test size(PI2) == (4, 2, 2)
        @test all(sum(PI2[i, s, :]) ≈ 1.0 for i in 1:size(PI2, 1), s in S)
    end

    @testset "Nearest Neighbor" begin
        belief_space = [1.0 0.0; 0.0 1.0]
        @test AggregationUtil.find_nearest_neighbor_index(belief_space, [0.8, 0.2]) == 1
        @test AggregationUtil.find_nearest_neighbor_index(belief_space, [0.1, 0.9]) == 2
        @test AggregationUtil.find_nearest_neighbor_belief(belief_space, [0.1, 0.9]) ≈ [0.0, 1.0]
    end

    @testset "Tensor Calculations" begin        
        n, num_states, num_actions_player_1, num_actions_player_2, num_observations = 1, 2, 1, 1, 2
        S, A1, A2, O = collect(1:num_states), collect(1:num_actions_player_1), collect(1:num_actions_player_2), collect(1:num_observations)

        agg_belief_space = AggregationUtil.generate_aggregate_belief_space(n, num_states)

        PI2 = ones(1, num_states, num_actions_player_2)

        R = zeros(num_actions_player_1, num_actions_player_2, num_states)
        R[1, 1, 1] = 100.0
        R[1, 1, 2] = 200.0

        belief_R = AggregationUtil.generate_aggregate_belief_reward_tensor(agg_belief_space, PI2, R)

        @test size(belief_R) == (1, 1, 2)
        @test belief_R[1, 1, 1] ≈ 100.0        
        @test belief_R[1, 1, 2] ≈ 200.0

        T = zeros(num_actions_player_1, num_actions_player_2, num_states, num_states)
        T[1, 1, 1, 1] = 1.0
        T[1, 1, 2, 2] = 1.0

        Z = zeros(num_actions_player_1, num_actions_player_2, num_states, num_observations)
        Z[1, 1, 1, 1] = 0.4
        Z[1, 1, 1, 2] = 0.6
        Z[1, 1, 2, 1] = 0.7
        Z[1, 1, 2, 2] = 0.3

        belief_T = AggregationUtil.generate_aggregate_belief_transition_operator(agg_belief_space, num_states, num_actions_player_1, num_actions_player_2, 
        PI2, num_observations, T, Z)
        
        @test belief_T[1, 1, 1, 1] ≈ 1.0
        @test belief_T[1, 1, 1, 2] ≈ 0.0        
        @test sum(belief_T[1, 1, 1, :]) ≈ 1.0
    end
end