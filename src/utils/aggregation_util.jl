
    function _integer_compositions(n::Int, k::Int)::Vector{Vector{Int}}
        return collect(multiexponents(k, n))
    end

    function generate_aggregate_belief_space(n::Int, belief_space_dimension::Int)::Matrix{Float64}
        combinations = _integer_compositions(n, belief_space_dimension)
        belief_points = [c ./ n for c in combinations]
        return hcat(belief_points...)'
    end

    function generate_aggregate_action_space_of_player_2(n::Int, num_a2::Int, num_states::Int)::Array{Float64,3}        
        combinations = _integer_compositions(n, num_a2)
        action_probabilities = [c ./ n for c in combinations]
        pi2_strategies = collect(Iterators.product(ntuple(_ -> action_probabilities, num_states)...))
        num_strategies = length(pi2_strategies)
        PI2 = Array{Float64,3}(undef, num_strategies, num_states, num_a2)
        for i in 1:num_strategies
            for s in 1:num_states
                PI2[i, s, :] = pi2_strategies[i][s]
            end
        end
        return PI2
    end

    function find_nearest_neighbor_belief(belief_space::Matrix{Float64}, target_belief::Vector{Float64})::Vector{Float64}
        distances = [norm(row - target_belief) for row in eachrow(belief_space)]
        nearest_index = argmin(distances)
        return belief_space[nearest_index, :]
    end

    function generate_aggregate_belief_reward_tensor(agg_belief_space::Matrix{Float64}, PI2::Array{Float64,3}, R::Array{Float64,3})::Array{Float64,3}
        @einsum belief_R[a, j, b] := agg_belief_space[b, s] * PI2[j, s, k] * R[a, k, s]
        return belief_R
    end

    function find_nearest_neighbor_index(belief_space::Matrix{Float64}, target_belief::Vector{Float64})::Int    
        distances = [norm(row - target_belief) for row in eachrow(belief_space)]    
        return argmin(distances)
    end

    function generate_aggregate_belief_transition_operator(agg_belief_space::Matrix{Float64}, num_states:: Int, num_a1:: Int, 
        num_a2:: Int, PI2::Array{Float64,3}, num_obs::Int, T::Array{Float64,4}, Z::Array{Float64,4})::Array{Float64,4}
        num_b = size(agg_belief_space, 1)        
        num_pi2 = size(PI2, 1)        
        belief_T = zeros(num_a1, num_pi2, num_b, num_b)    
        for a1_idx in 1:num_a1
            for pi2_idx in 1:num_pi2
                pi2 = PI2[pi2_idx, :, :]
                for b1_idx in 1:num_b
                    b1 = agg_belief_space[b1_idx, :]
                    for o_idx in 1:num_obs
                        prob_o = sum(
                            Z[a1_idx, a2, s_prime, o_idx] * T[a1_idx, a2, s, s_prime] * b1[s] * pi2[s, a2]
                            for s in 1:num_states, s_prime in 1:num_states, a2 in 1:num_a2
                        )
                        isapprox(prob_o, 0) && continue
                        b_prime = OsPosgUtil.next_belief(o_idx, a1_idx, b1, pi2, num_states, num_a2, Z, T)
                        b2_idx = find_nearest_neighbor_index(agg_belief_space, b_prime)
                        belief_T[a1_idx, pi2_idx, b1_idx, b2_idx] += prob_o
                    end
                end
            end
        end
        return belief_T
    end


    function generate_aggregate_belief_transition_operator_ann(agg_belief_space::Matrix{Float64}, num_states::Int, num_a1::Int, 
        num_a2::Int, PI2::Array{Float64,3}, num_obs::Int, T::Array{Float64,4}, Z::Array{Float64,4})::Array{Float64,4}
        num_b = size(agg_belief_space, 1)        
        num_pi2 = size(PI2, 1)
        belief_T = zeros(num_a1, num_pi2, num_b, num_b)
        kdtree = KDTree(agg_belief_space')
        for a1_idx in 1:num_a1
            for pi2_idx in 1:num_pi2
                pi2 = PI2[pi2_idx, :, :]
                for b1_idx in 1:num_b
                    b1 = agg_belief_space[b1_idx, :]
                    for o_idx in 1:num_obs
                        prob_o = sum(
                            Z[a1_idx, a2, s_prime, o_idx] * T[a1_idx, a2, s, s_prime] * b1[s] * pi2[s, a2]
                            for s in 1:num_states, s_prime in 1:num_states, a2 in 1:num_a2
                        )
                        isapprox(prob_o, 0) && continue
                        b_prime = OsPosgUtil.next_belief(o_idx, a1_idx, b1, pi2, num_states, num_a2, Z, T)
                        idxs, _ = knn(kdtree, b_prime, 1)
                        b2_idx = idxs[1]
                        belief_T[a1_idx, pi2_idx, b1_idx, b2_idx] += prob_o
                    end
                end
            end
        end
        return belief_T
    end

    function generate_aggregate_belief_transition_operator_sparse_ann(agg_belief_space::Matrix{Float64}, num_states::Int, num_a1::Int, 
        num_a2::Int, PI2::Array{Float64,3}, num_obs::Int, T::Array{Float64,4}, Z::Array{Float64,4})::SparseMatrixCSC{Float64, Int}

        num_b = size(agg_belief_space, 1)        
        num_pi2 = size(PI2, 1)        
        kdtree = KDTree(agg_belief_space')

        #Build coordinates for a sparse matrix ---
        I = Int[] # row indices
        J = Int[] # column indices
        V = Float64[] # values

        # Helper function to flatten the first three dimensions into a single row index
        row_idx(a1, p2, b1) = a1 + (p2 - 1) * num_a1 + (b1 - 1) * num_a1 * num_pi2

        for a1_idx in 1:num_a1
            for pi2_idx in 1:num_pi2
                pi2 = PI2[pi2_idx, :, :]
                for b1_idx in 1:num_b
                    b1 = agg_belief_space[b1_idx, :]
                    for o_idx in 1:num_obs                    
                        prob_o = sum(
                            Z[a1_idx, a2, s_prime, o_idx] * T[a1_idx, a2, s, s_prime] * b1[s] * pi2[s, a2]
                            for s in 1:num_states, s_prime in 1:num_states, a2 in 1:num_a2
                        )
                        # Only store non-zero transition probabilities
                        if !isapprox(prob_o, 0)
                            b_prime = OsPosgUtil.next_belief(o_idx, a1_idx, b1, pi2, num_states, num_a2, Z, T)

                            # Use the fast KD-tree to find the nearest neighbor's index
                            idxs, _ = knn(kdtree, b_prime, 1)
                            b2_idx = idxs[1]

                            # Store the coordinates and value for the sparse matrix
                            push!(I, row_idx(a1_idx, pi2_idx, b1_idx))
                            push!(J, b2_idx)
                            push!(V, prob_o)
                        end
                    end
                end
            end
        end    
        num_rows = num_a1 * num_pi2 * num_b
        num_cols = num_b
        return sparse(I, J, V, num_rows, num_cols)
    end