
    function compute_matrix_game_value(A::Matrix{Float64}; maximizer::Bool=true)::Tuple{Float64,Vector{Float64}}
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

    function auxillary_game(V::Vector{Float64}, gamma::Float64, b_idx::Int, num_a1::Int, num_pi2::Int,
        R_b::Array{Float64,3}, T_b::Array{Float64,4})::Matrix{Float64}        
        A = zeros(num_a1, num_pi2)
        for a1 in 1:num_a1
            for i in 1:num_pi2
                immediate_reward = R_b[a1, i, b_idx]
                expected_future_reward = dot(T_b[a1, i, b_idx, :], V) * gamma
                A[a1, i] = immediate_reward + expected_future_reward
            end
        end
        return A
    end

    function shapley_iteration(B_n::Matrix{Float64}, max_iterations::Int, gamma::Float64, num_a1::Int,
        num_pi2::Int, R_b::Array{Float64,3}, T_b::Array{Float64,4}; delta_threshold::Float64=0.1,
        log_every::Int=1, verbose::Bool=true)::Tuple{Vector{Float64},Matrix{Float64},Matrix{Float64},Array{Float64,3},Vector{Float64}}

        deltas = Float64[]
        num_beliefs = size(B_n, 1)
        V = zeros(num_beliefs)
        V_new = similar(V)

        for i in 1:max_iterations
            auxillary_games = [auxillary_game(V, gamma, b_idx, num_a1, num_pi2, R_b, T_b) for b_idx in 1:num_beliefs]

            for b_idx in 1:num_beliefs
                value, _ = compute_matrix_game_value(auxillary_games[b_idx], maximizer=true)
                V_new[b_idx] = value
            end

            delta = norm(V - V_new, 1)
            push!(deltas, delta)

            V .= V_new

            if i % log_every == 0 && i > 0 && verbose
                println("[Shapley iteration] i:$i, delta: $delta")
            end

            if delta <= delta_threshold && verbose
                println("Convergence reached at iteration $i.")
                break
            end
        end

        maximin_strategies = Vector{Float64}[]
        minimax_strategies = Vector{Float64}[]
        final_aux_games = [auxillary_game(V, gamma, b_idx, num_a1, num_pi2, R_b, T_b) for b_idx in 1:num_beliefs]

        for A in final_aux_games
            _, maximin_s = compute_matrix_game_value(A, maximizer=true)
            _, minimax_s = compute_matrix_game_value(A, maximizer=false)
            push!(maximin_strategies, maximin_s)
            push!(minimax_strategies, minimax_s)
        end

        maximin_matrix = hcat(maximin_strategies...)'
        minimax_matrix = hcat(minimax_strategies...)'
        games_tensor = reduce((x, y) -> cat(x, y, dims=3), final_aux_games)

        return V, maximin_matrix, minimax_matrix, games_tensor, deltas
    end


    function auxillary_game_sparse(V::Vector{Float64}, gamma::Float64, b_idx::Int, num_a1::Int, num_pi2::Int, 
        R_b::Array{Float64,3}, T_b_sparse::SparseMatrixCSC{Float64, Int})::Matrix{Float64}    
        A = zeros(num_a1, num_pi2)

        # This helper function maps the 3D index (a1, pi2, b1) to the
        # correct single row index in the flattened sparse matrix.
        # It must be identical to the one used to create the sparse matrix.
        row_idx(a1, p2, b1) = a1 + (p2 - 1) * num_a1 + (b1 - 1) * num_a1 * num_pi2

        for a1 in 1:num_a1
            for i in 1:num_pi2
                immediate_reward = R_b[a1, i, b_idx]

                # Find the row in the sparse matrix corresponding to this state
                current_row_index = row_idx(a1, i, b_idx)

                # Extract the sparse row vector. This is a very fast operation.
                transition_probabilities_row = T_b_sparse[current_row_index, :]

                # The `dot` product is optimized for sparse vectors. It will
                # only iterate over the non-zero elements, providing speedup.
                expected_future_reward = dot(transition_probabilities_row, V) * gamma

                A[a1, i] = immediate_reward + expected_future_reward
            end
        end
        return A
    end

    function shapley_iteration_sparse(B_n::Matrix{Float64}, max_iterations::Int, gamma::Float64, num_a1::Int, 
        num_pi2::Int, R_b::Array{Float64,3}, T_b_sparse::SparseMatrixCSC{Float64, Int}; 
        delta_threshold::Float64=0.1, log_every::Int=1, verbose::Bool=true)::Tuple{Vector{Float64}, Matrix{Float64}, Matrix{Float64}, Array{Float64, 3}, Vector{Float64}}

        deltas = Float64[]
        num_beliefs = size(B_n, 1)
        V = zeros(num_beliefs)
        V_new = similar(V)

        for i in 1:max_iterations
            # This now calls the sparse-aware version of the function
            auxillary_games = [auxillary_game_sparse(V, gamma, b_idx, num_a1, num_pi2, R_b, T_b_sparse) for b_idx in 1:num_beliefs]

            for b_idx in 1:num_beliefs
                value, _ = compute_matrix_game_value(auxillary_games[b_idx], maximizer=true)
                V_new[b_idx] = value
            end

            delta = norm(V - V_new, 1)
            push!(deltas, delta)

            V .= V_new

            if i % log_every == 0 && i > 0 && verbose
                println("[Shapley iteration] i:$i, delta: $delta")
            end

            if delta <= delta_threshold && verbose
                println("Convergence reached at iteration $i.")
                break
            end
        end

        maximin_strategies = Vector{Float64}[]
        minimax_strategies = Vector{Float64}[]
        # This also calls the sparse-aware version
        final_aux_games = [auxillary_game_sparse(V, gamma, b_idx, num_a1, num_pi2, R_b, T_b_sparse) for b_idx in 1:num_beliefs]

        for A in final_aux_games
            _, maximin_s = compute_matrix_game_value(A, maximizer=true)
            _, minimax_s = compute_matrix_game_value(A, maximizer=false)
            push!(maximin_strategies, maximin_s)
            push!(minimax_strategies, minimax_s)
        end

        maximin_matrix = hcat(maximin_strategies...)'
        minimax_matrix = hcat(minimax_strategies...)'
        games_tensor = reduce((x, y) -> cat(x, y, dims=3), final_aux_games)

        return V, maximin_matrix, minimax_matrix, games_tensor, deltas
    end
