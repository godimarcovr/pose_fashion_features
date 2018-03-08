function [sim_mat, partial_sims, color_partial_sims] = compute_sim_mat_fn(n_components, n_dimensions, samples,color_samples,color_lambda,comp_filter,symm,overlaps)
index_cache = [];
for comp_i=1:n_components
    index_cache = [index_cache ones(1, n_dimensions) .* comp_i];
end
color_index_cache = [];
for comp_i=1:n_components
    color_index_cache = [color_index_cache ones(1, 11) .* comp_i];
end

if numel(comp_filter) > 0
    samples = samples(:, index_cache == comp_filter);
    color_samples = color_samples(:, color_index_cache == comp_filter);
    n_components = 1;
    overlaps = overlaps(:, comp_filter);
end



partial_sims = zeros(n_components, size(samples, 1), size(samples, 1));
color_partial_sims = zeros(n_components, size(color_samples, 1), size(color_samples, 1));
if numel(comp_filter) == 0 && symm
    [~, partial_sims] = HP_feats_similarity_fn(samples,n_dimensions);
    [~, color_partial_sims] = HP_feats_similarity_fn(color_samples,11);
else
    for comp_i=1:n_components
        norms = vecnorm(samples(:, index_cache == comp_i),2,2);
        norms2 = norms * norms';
        partial_sims(comp_i,:, :) = samples(:, index_cache == comp_i) * samples(:, index_cache == comp_i)';
        tmp = reshape(partial_sims(comp_i,:, :), size(samples, 1), size(samples, 1)) ./ norms2;
        partial_sims(comp_i,:, :) = reshape(tmp, 1, size(samples, 1), size(samples, 1));

        color_partial_sims(comp_i,:, :) = 1 - squareform(pdist(color_samples(:, color_index_cache == comp_i), 'cosine'));
    end
end
partial_sims = (partial_sims ./ 2) + 0.5;

if numel(overlaps) > 0
%     overlaps = (overlaps ./ sum(overlaps, 2)) .* n_components;
%     overlaps(isnan(overlaps)) = 1.0;
    for comp_i=1:n_components
        %soglia sopra 0.5 porta a 1.0??
        overlap_penalty = overlaps(:, comp_i) * overlaps(:, comp_i)';
        overlap_penalty = tanh(3 * overlap_penalty);
        
        tmp = reshape(partial_sims(comp_i,:, :), size(samples, 1), size(samples, 1)) .* overlap_penalty;
        partial_sims(comp_i,:, :) = reshape(tmp, 1, size(samples, 1), size(samples, 1));
        tmp = reshape(color_partial_sims(comp_i,:, :), size(color_samples, 1), size(color_samples, 1)) .* overlap_penalty;
        color_partial_sims(comp_i,:, :) = reshape(tmp, 1, size(color_samples, 1), size(color_samples, 1));
    end
    
%     partial_sims = partial_sims .* overlaps;
%     overlap_factors = overlaps * overlaps';
%     partial_sims = partial_sims .* overlap_factors;
end 

tmp_pc = prctile(partial_sims(:), 99.9);
partial_sims = (tanh(25*(partial_sims - 0.5)) ./ 2 + 0.5) ./ tmp_pc;
% partial_sims = partial_sims ./ tmp_pc;
partial_sims(partial_sims>1) = 1;

sim_mat = reshape(mean(partial_sims, 1), size(samples, 1), size(samples, 1));
color_sim_mat = reshape(mean(color_partial_sims, 1), size(color_samples, 1), size(color_samples, 1));

% color_sim_mat = 1 - squareform(pdist(color_samples,'cosine')); %come fare i componenti? Media pesata?
sim_mat = color_lambda * color_sim_mat + (1 - color_lambda) * sim_mat;



end

