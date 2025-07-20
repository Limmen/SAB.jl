
@testset "os_posg_util Tests" begin

    # --- Test Environment Setup ---
    num_states = 3
    num_actions_player_1 = 2
    num_actions_player_2 = 2 # Number of actions for player 2
    num_observations = 2 # Number of observations

    S = collect(1:num_states)
    A1 = collect(1:num_actions_player_1)
    A2 = collect(1:num_actions_player_2)
    O = collect(1:num_observations)

    # --- Mock Data ---
    b_uniform = ones(num_states) ./ num_states
    b_deterministic = [0.0, 1.0, 0.0]
    pi2_uniform = ones(num_states, num_actions_player_2) ./ num_actions_player_2
    pi2_deterministic = [1.0 0.0; 0.0 1.0; 1.0 0.0]
    T = ones(num_actions_player_1, num_actions_player_2, num_states, num_states) ./ num_states
    T[1, 1, 1, :] = [0.0, 0.0, 1.0]
    Z = ones(num_actions_player_1, num_actions_player_2, num_states, num_observations) ./ num_observations
    Z[1, 1, 3, :] = [1.0, 0.0]

    # --- Test Cases ---

    @testset "Sampling Functions" begin
        @testset "sample_initial_state" begin
            @test OsPosgUtil.sample_initial_state(b_deterministic) == 2
            s = OsPosgUtil.sample_initial_state(b_uniform)
            @test s isa Integer
            @test s in S
        end

        @testset "sample_player_2_action" begin
            @test OsPosgUtil.sample_player_2_action(pi2_deterministic, 2) == 2
            a2 = OsPosgUtil.sample_player_2_action(pi2_uniform, 1)
            @test a2 isa Integer
            @test a2 in A2
        end

        @testset "sample_next_state" begin
            @test OsPosgUtil.sample_next_state(T, 1, 1, 1) == 3
            s_prime = OsPosgUtil.sample_next_state(T, 2, 2, 2)
            @test s_prime isa Integer
            @test s_prime in S
        end

        @testset "sample_next_observation" begin
            @test OsPosgUtil.sample_next_observation(Z, 3, 1, 1) == 1
            o = OsPosgUtil.sample_next_observation(Z, 1, 2, 2)
            @test o isa Integer
            @test o in O
        end
    end

    @testset "Belief Update Functions" begin
        # --- Setup for a fully deterministic belief update test ---
        num_states_b = 2        
        num_a2_b = 1

        b_b = [0.5, 0.5]
        pi2_b = ones(num_states_b, 1)

        T_b = zeros(1, 1, num_states_b, num_states_b)
        T_b[1, 1, 1, 1] = 1.0
        T_b[1, 1, 2, 2] = 1.0

        Z_b = zeros(1, 1, num_states_b, num_states_b)
        Z_b[1, 1, 1, 1] = 1.0
        Z_b[1, 1, 2, 2] = 1.0

        @testset "bayes_filter" begin
            b_prime_s1 = OsPosgUtil.bayes_filter(1, 1, 1, b_b, pi2_b, num_states_b, num_a2_b, Z_b, T_b)
            @test b_prime_s1 ≈ 1.0

            b_prime_s2 = OsPosgUtil.bayes_filter(2, 1, 1, b_b, pi2_b, num_states_b, num_a2_b, Z_b, T_b)
            @test b_prime_s2 ≈ 0.0

            b_impossible = [1.0, 0.0]
            b_prime_impossible = OsPosgUtil.bayes_filter(1, 2, 1, b_impossible, pi2_b, num_states_b, num_a2_b, Z_b, T_b)
            @test b_prime_impossible == 0.0
        end

        @testset "next_belief" begin
            b_prime_vec = OsPosgUtil.next_belief(1, 1, b_b, pi2_b, num_states_b, num_a2_b, Z_b, T_b)
            @test b_prime_vec isa Vector{Float64}
            @test b_prime_vec ≈ [1.0, 0.0]
            @test sum(b_prime_vec) ≈ 1.0
        end
    end
end