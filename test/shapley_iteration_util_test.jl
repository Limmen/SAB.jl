@testset "shapley_iteration_util tests" begin
    n = 2
    m = 2
    num_obs = 2
    gamma = 0.8
    delta_threshold = 0.5
    max_iterations = 10

    b1 = StoppingGame.b1()
    A1 = StoppingGame.player_1_actions()
    A2 = StoppingGame.player_2_actions()
    S = StoppingGame.state_space()
    O = StoppingGame.observation_space(num_obs)
    Z = StoppingGame.observation_tensor(num_obs)
    T = StoppingGame.transition_tensor()
    R = StoppingGame.reward_tensor()

    B_n = AggregationUtil.generate_aggregate_belief_space(n, length(S))
    PI2 = AggregationUtil.generate_aggregate_action_space_of_player_2(m, length(A2), length(S))
    T_b = AggregationUtil.generate_aggregate_belief_transition_operator_sparse_ann(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
    R_b = AggregationUtil.generate_aggregate_belief_reward_tensor(B_n, PI2, R)

    V, maximin_strategies, minimax_strategies, auxillary_games, deltas = ShapleyIteration.shapley_iteration_sparse(
    B_n, max_iterations, gamma, length(A1), size(PI2, 1), R_b, T_b; 
    delta_threshold=delta_threshold, log_every=1, verbose=false)
    @test length(V) == size(B_n, 1)

    T_b = AggregationUtil.generate_aggregate_belief_transition_operator_ann(B_n, length(S), length(A1), length(A2), PI2, length(O), T, Z)
    V, maximin_strategies, minimax_strategies, auxillary_games, deltas = ShapleyIteration.shapley_iteration(
    B_n, max_iterations, gamma, length(A1), size(PI2, 1), R_b, T_b; 
    delta_threshold=delta_threshold, log_every=1, verbose=false)    
    @test length(V) == size(B_n, 1)

end