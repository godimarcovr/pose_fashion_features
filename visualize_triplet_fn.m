function visualize_triplet_fn(i,bestind, worstind, partial_sims_best, partial_sims_worst, color_partial_sims_best, color_partial_sims_worst, dataset, imsize, jpatch_w, color_lambda, overlaps, n_components, set, net, encoder)


comp_scores_best = partial_sims_best * (1 - color_lambda)  + color_lambda * color_partial_sims_best;
comp_scores_worst = partial_sims_worst * (1 - color_lambda)  + color_lambda * color_partial_sims_worst;
comps = [];

%*****
subplot(1,3,2)
coords_i = compute_coords(i, dataset, imsize, jpatch_w, n_components, set);
subplot(1,3,1)
coords_best = compute_coords(bestind, dataset, imsize, jpatch_w, n_components, set);
subplot(1,3,3)
coords_worst = compute_coords(worstind, dataset, imsize, jpatch_w, n_components, set);
%****

subplot(1,3,1)
comps_best = draw_areas(comps, partial_sims_best, color_partial_sims_best, comp_scores_best, coords_best, overlaps, bestind);

subplot(1,3,2)
draw_areas(comps_best, partial_sims_best, color_partial_sims_best, comp_scores_best, coords_i, overlaps, i, true);

subplot(1,3,3)
draw_areas(comps_best, partial_sims_worst, color_partial_sims_worst, comp_scores_worst, coords_worst, overlaps, worstind);

%*********
[~,~,button]=ginput(1);
switch button
  case 3 %left
      print(['similarities_' num2str(i) '_' num2str(bestind) '_' num2str(worstind)],'-dpng', '-r0');
      [partial_sims_64k_best, partial_sims_64k_worst] = compute_partial_sims_64k(i, bestind, worstind, dataset, jpatch_w, set, net, encoder,imsize);
      show_64k_tripletfigure(i,bestind, worstind, partial_sims_64k_best, partial_sims_64k_worst, color_partial_sims_best, color_partial_sims_worst,...
                                dataset, imsize, jpatch_w, overlaps, n_components, set, comps_best);
      print(['similarities64k_' num2str(i) '_' num2str(bestind) '_' num2str(worstind)],'-dpng', '-r0');

  case 1 %right
      1;
end

end

function show_64k_tripletfigure(i,bestind, worstind, partial_sims_best, partial_sims_worst, color_partial_sims_best, color_partial_sims_worst,...
                                dataset, imsize, jpatch_w, overlaps, n_components, set, comps)
    figure
    comp_scores_best = partial_sims_best;
    comp_scores_worst = partial_sims_worst;

    %*****
    subplot(1,3,2)
    coords_i = compute_coords(i, dataset, imsize, jpatch_w, n_components, set);
    subplot(1,3,1)
    coords_best = compute_coords(bestind, dataset, imsize, jpatch_w, n_components, set);
    subplot(1,3,3)
    coords_worst = compute_coords(worstind, dataset, imsize, jpatch_w, n_components, set);
    %****

    subplot(1,3,1)
    comps_best = draw_areas(comps, partial_sims_best, color_partial_sims_best, comp_scores_best, coords_best, overlaps, bestind);

    subplot(1,3,2)
    draw_areas(comps_best, partial_sims_best, color_partial_sims_best, comp_scores_best, coords_i, overlaps, i, true);

    subplot(1,3,3)
    draw_areas(comps_best, partial_sims_worst, color_partial_sims_worst, comp_scores_worst, coords_worst, overlaps, worstind);

end

function [partial_sims_64k_best, partial_sims_64k_worst] = compute_partial_sims_64k(i, bestind, worstind, dataset, jpatch_w, set, net, encoder,imsize)
    feats_i = compute_64kfeatures(i, dataset, imsize, jpatch_w, set, net, encoder);
    feats_best = compute_64kfeatures(bestind, dataset, imsize, jpatch_w, set, net, encoder);
    feats_worst = compute_64kfeatures(worstind, dataset, imsize, jpatch_w, set, net, encoder);
    partial_sims_64k_best = zeros(numel(feats_i), 1);
    partial_sims_64k_worst = zeros(numel(feats_i), 1);
    for i=1:numel(feats_i)
        partial_sims_64k_best(i) = pdist( [feats_i{i}.feats; feats_best{i}.feats]);
        partial_sims_64k_worst(i) = pdist([feats_i{i}.feats; feats_worst{i}.feats]);
    end
end


function feats = compute_64kfeatures(i, dataset, imsize, jpatch_w, set, net, encoder)
    tmp_path = dataset.(set).all{i};
    tmp_pose_info = dataset.(set).pose_info.all{i};
    tmp_parsing_path = dataset.(set).parsing.all{i};
    feats = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, encoder, tmp_parsing_path);
end

function coords = compute_coords(i, dataset, imsize, jpatch_w, n_components, set)
    img = imread(dataset.(set).all{i});
    pose_info = dataset.(set).pose_info.all{i};
    sc_x = size(img, 2) / imsize(2);
    sc_y = size(img, 1) / imsize(1);
    img = imresize(img, imsize);
    imshow(img);
    hold on
    title(num2str(dataset.(set).all_labels(i)));
    jpatch_size = jpatch_w * (pose_info.width / sc_x);

    coords = [pose_info.coords floor((pose_info.segments(1:2, :) + pose_info.segments(3:4, :)) ./ 2)];
    coords = [coords./[sc_x; sc_y]; ones(2, n_components).*jpatch_size];
    coords(1:2, :) = coords(1:2, :) - floor(jpatch_size/2);
    coords(1:2, coords<0) = 0;
    hold off
end

function comps = draw_areas(comps, partial_sims, color_partial_sims, comp_scores, coords, overlaps, i, isnotext)
    if ~exist('isnotext', 'var')
        isnotext = false;
    end
    hold on
    if numel(comps) == 0
        for comp_i=1:numel(comp_scores)
            comp_score1 = partial_sims(comp_i);
            if comp_score1 >= prctile(partial_sims,90)
                if all(rectint(coords(:, comps)',coords(:, comp_i)') == 0)
                    comps = [comps comp_i];
                end
            end
        end
    end
    for comp_i=comps
        comp_score1 = partial_sims(comp_i);
        comp_score = comp_scores(comp_i);
        
    %         if comp_score > prctile(comp_scores,90) | comp_score < prctile(comp_scores,10)
        rectangle('Position',coords(:,comp_i)','EdgeColor', 'red');
        if ~isnotext
            text('Position',coords(1:2 ,comp_i)' + [0 6],'string',sprintf('%.2f', comp_score1), 'Color', 'red')
            text('Position',coords(1:2 ,comp_i)' + [16 6],'string',sprintf('%.2f', color_partial_sims(comp_i)), 'Color', 'cyan')
            text('Position',coords(1:2 ,comp_i)' + [16 12],'string',sprintf('%.2f', comp_score), 'Color', 'yellow')
        end
        text('Position',coords(1:2 ,comp_i)' + [0 12],'string',sprintf('%.2f', overlaps(i, comp_i)), 'Color', 'green')
    %         end
    end
    hold off
end

