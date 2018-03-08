function sim_mat = HP_compare_LR_fn(samples,n_dimensions)
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
    for joints=equivs'
        comp_i
        j1 = joints(1);
        j2 = joints(2);
        sims_vals = samples(:, index_cache == j1) * samples(:, index_cache == j2)';
    end
    sim_mat = sum(partial_sims, 1);
end

