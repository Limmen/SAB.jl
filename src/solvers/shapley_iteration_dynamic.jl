
function compute_matrix_game_value_dynamic(A::Matrix{Float64}; maximizer::Bool=true)::Tuple{Float64,Vector{Float64}}
    num_rows, num_cols = size(A)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, v)
    if maximizer
        @variable(model, s[1:num_rows] >= 0)
        @objective(model, Max, v)
        @constraint(model, [j = 1:num_cols], sum(s[i] * A[i, j] for i = 1:num_rows) >= v)
    else
        @variable(model, s[1:num_cols] >= 0)
        @objective(model, Min, v)
        @constraint(model, [i = 1:num_rows], sum(s[j] * A[i, j] for j = 1:num_cols) <= v)
    end

    @constraint(model, sum(s) == 1)
    optimize!(model)
    return value(v), value.(s)
end

const MAX_P2_POLICIES = 10000

function generate_pure_policies_for_belief(
    b::Vector{Float64},
    num_a2::Int,
    num_states::Int
)::Tuple{Vector{Vector{Int}},Vector{Int}}
    active_states_indices = findall(x -> !isapprox(x, 0), b)
    num_active_states = length(active_states_indices)
    num_policies_to_generate = num_a2^num_active_states
    if num_policies_to_generate > MAX_P2_POLICIES
        @warn "Policy generation for belief requested $(num_policies_to_generate) policies, exceeding limit of $(MAX_P2_POLICIES). Belief has $(num_active_states) active states. Truncating."
        num_policies_to_generate = MAX_P2_POLICIES
    end
    policy_action_combinations = IterTools.product(ntuple(_ -> 1:num_a2, num_active_states)...)
    policies = Vector{Vector{Int}}()    
    for (i, combo) in enumerate(policy_action_combinations)
        if i > num_policies_to_generate
            break
        end
        full_policy = ones(Int, num_states)
        full_policy[active_states_indices] .= combo
        push!(policies, full_policy)
    end

    return policies, active_states_indices
end

function compute_auxiliary_game_fully_dynamic(
    V::Vector{Float64},
    gamma::Float64,
    b_idx::Int,
    b1::Vector{Float64},
    pi2_pure_policies::Vector{Vector{Int}}, 
    active_states_indices::Vector{Int},
    agg_belief_space::Matrix{Float64},
    R::Array{Float64,3}, T::Array{Float64,4}, Z::Array{Float64,4},
    kdtree::KDTree,
    num_a1::Int, num_a2::Int, num_states::Int, num_obs::Int
)::Matrix{Float64}

    num_pi2_dynamic = length(pi2_pure_policies)
    A = zeros(num_a1, num_pi2_dynamic)

    # This matrix is needed to conform to the `next_belief` utility's API
    pi2_stochastic_matrix = zeros(num_states, num_a2)

    for a1_idx in 1:num_a1
        for pi2_policy_idx in 1:num_pi2_dynamic
            pi2_policy = pi2_pure_policies[pi2_policy_idx] # This is a vector: policy[s] -> a2

            # 1. Compute Immediate Reward
            # The sum over a2 is removed, as the action is now deterministic for each state.
            immediate_reward = sum(
                b1[s] * R[a1_idx, pi2_policy[s], s]
                for s in active_states_indices # Only need to sum over active states
            )

            # 2. Compute Expected Future Reward
            expected_future_reward = 0.0

            # Create a stochastic-like policy matrix to pass to the utility function
            fill!(pi2_stochastic_matrix, 0.0)
            for s in 1:num_states
                if pi2_policy[s] > 0
                    pi2_stochastic_matrix[s, pi2_policy[s]] = 1.0
                end
            end

            for o_idx in 1:num_obs
                # Calculate prob_o using the deterministic policy
                prob_o = sum(
                    b1[s] * T[a1_idx, pi2_policy[s], s, s_prime] * Z[a1_idx, pi2_policy[s], s_prime, o_idx]
                    for s in active_states_indices, s_prime in 1:num_states
                )

                if isapprox(prob_o, 0)
                    continue
                end

                b_prime = OsPosgUtil.next_belief(o_idx, a1_idx, b1, pi2_stochastic_matrix, num_states, num_a2, Z, T)
                nearest_b_prime_idx, _ = knn(kdtree, b_prime, 1)
                expected_future_reward += prob_o * V[nearest_b_prime_idx[1]]
            end

            A[a1_idx, pi2_policy_idx] = immediate_reward + gamma * expected_future_reward
        end
    end
    return A
end

function shapley_iteration_dynamic(
    agg_belief_space::Matrix{Float64},
    num_a2::Int, # We only need the number of actions for P2
    R::Array{Float64,3}, T::Array{Float64,4}, Z::Array{Float64,4},
    max_iterations::Int, gamma::Float64, V_star::Array{Float64}, B_eval::Matrix{Float64}, threshold::Float64;
    delta_threshold::Float64=0.1, log_every::Int=1, verbose::Bool=true
)
    num_beliefs = size(agg_belief_space, 1)
    num_states = size(agg_belief_space, 2)
    num_a1 = size(R, 1)
    num_obs = size(Z, 4)

    kdtree = KDTree(agg_belief_space')
    V = zeros(num_beliefs)
    V_new = similar(V)

    start = time()
    initial_delta = -1
    prev_delta = 1000

    println("Starting Shapley iteration...")
    for i in 1:max_iterations
        for b_idx in 1:num_beliefs
            b1 = agg_belief_space[b_idx, :]
            pi2_pure_policies, active_indices = generate_pure_policies_for_belief(b1, num_a2, num_states)

            if isempty(pi2_pure_policies)
                V_new[b_idx] = 0.0
                continue
            end

            aux_game = compute_auxiliary_game_fully_dynamic(
                V, gamma, b_idx, b1, pi2_pure_policies, active_indices,
                agg_belief_space, R, T, Z, kdtree,
                num_a1, num_a2, num_states, num_obs
            )

            value, _ = compute_matrix_game_value(aux_game, maximizer=true)
            V_new[b_idx] = value
        end

        delta = norm(V - V_new, 1)
        if initial_delta == -1
            initial_delta = delta
        end
        V .= V_new
        V_eval = []
        for i in 1:size(B_eval, 1)
            b = B_eval[i, :]
            nearest_belief_index = AggregationUtil.find_nearest_neighbor_index(agg_belief_space, b)
            value = V[nearest_belief_index]
            push!(V_eval, value)
        end

        if i % log_every == 0 && verbose
            println("[Shapley Iteration] i:$i, delta: $delta, approx error: $(norm(V_eval - V_star, Inf)), time: $(round(time()-start, digits=1)) seconds, $((delta)/initial_delta)")
        end
        prev_delta = delta
        if (delta / initial_delta) <= threshold
            println("Convergence reached.")
            break
        end

        if delta <= delta_threshold && verbose
            println("Convergence reached at iteration $i.")
            break
        end
    end

    return V
end

