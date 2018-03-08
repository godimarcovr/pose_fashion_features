function [sim_mat, partial_sims] = HP_feats_similarity_fn(samples,n_dimensions)
    n_joints = 16;
    n_segms = 15;
    n_components = n_joints + n_segms;
    joint_equivs = [1 6; 2 5; 3 4; 11 16; 12 15; 13 14];
    segm_equivs = [1 3; 2 4; 5 6; 10 15; 11 14; 12 13];
    segm_equivs = segm_equivs + n_joints;
    equivs = [joint_equivs; segm_equivs];
    
%     sim_mat = zeros(size(samples, 1));
    index_cache = [];
    for comp_i=1:n_components
        index_cache = [index_cache ones(1, n_dimensions) .* comp_i];
    end
    
    partial_sims = zeros(n_components, size(samples, 1), size(samples, 1));
    for comp_i=1:n_components
        if n_dimensions == 11
            partial_sims(comp_i,:, :) = 1 - squareform(pdist(samples(:, index_cache == comp_i), 'cosine'));
        else
            comp_i
            norms = vecnorm(samples(:, index_cache == comp_i),2,2);
            norms2 = norms * norms';
            partial_sims(comp_i,:, :) = samples(:, index_cache == comp_i) * samples(:, index_cache == comp_i)';
            tmp_mat = reshape(partial_sims(comp_i,:, :), size(samples, 1), size(samples, 1)) ./ norms2;
            partial_sims(comp_i,:, :) = reshape(tmp_mat, 1, size(samples, 1), size(samples, 1));
        end
    end
    for joints=equivs'
        j1 = joints(1);
        j1
        j2 = joints(2);
        if n_dimensions == 11
            sim_j1j2 = 1 - pdist2(samples(:, index_cache == comp_i),samples(:, index_cache == j2), 'cosine');
            sim_j2j1 = sim_j1j2';
        else
            sim_j1j2 = samples(:, index_cache == j1) * samples(:, index_cache == j2)';
            norms1 = vecnorm(samples(:, index_cache == j1),2,2);
            norms2 = vecnorm(samples(:, index_cache == j2),2,2);
            norms2 = norms1 * norms2';
            sim_j1j2 = sim_j1j2 ./ norms2;
            sim_j2j1 = sim_j1j2';
        end
        
        sims_vals = [partial_sims(j1, :, :); partial_sims(j2, :, :);...
            reshape(sim_j1j2, 1, size(partial_sims(j1, :, :), 2), size(partial_sims(j1, :, :), 3));...
            reshape(sim_j2j1, 1, size(partial_sims(j1, :, :), 2), size(partial_sims(j1, :, :), 3))];
        partial_sims(j1, :, :) = mean(sims_vals);
        partial_sims(j2, :, :) = mean(sims_vals);
    end
    sim_mat = sum(partial_sims, 1);
end

