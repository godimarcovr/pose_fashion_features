function visualize_similarities_fn(i,bestind, partial_sims, color_partial_sims, dataset, imsize, jpatch_w, color_lambda, overlaps, n_components, set)

comp_scores = partial_sims * (1 - color_lambda)  + color_lambda * color_partial_sims;
comps = [1 2 3 8];

%*****
subplot(1,2,1)

coords = compute_coords(i, dataset, imsize, jpatch_w, n_components, set);

draw_areas(comps, partial_sims, color_partial_sims, comp_scores, coords, overlaps, i);

%****
subplot(1,2,2)
coords = compute_coords(bestind, dataset, imsize, jpatch_w, n_components, set);

draw_areas(comps, partial_sims, color_partial_sims, comp_scores, coords, overlaps, bestind);

end



function coords = compute_coords(i, dataset, imsize, jpatch_w, n_components, set)
    img = imread(dataset.(set).all{i});
    pose_info = dataset.(set).pose_info.all{i};
    sc_x = size(img, 2) / imsize(2);
    sc_y = size(img, 1) / imsize(1);
    img = imresize(img, imsize);
    imshow(img);
    title(num2str(dataset.(set).all_labels(i)));
    jpatch_size = jpatch_w * (pose_info.width / sc_x);

    hold on
    coords = [pose_info.coords floor((pose_info.segments(1:2, :) + pose_info.segments(3:4, :)) ./ 2)];
    coords = [coords./[sc_x; sc_y]; ones(2, n_components).*jpatch_size];
    coords(1:2, :) = coords(1:2, :) - floor(jpatch_size/2);
    coords(1:2, coords<0) = 0;
end

function draw_areas(comps, partial_sims, color_partial_sims, comp_scores, coords, overlaps, i)
    hold on
    for comp_i=comps
        comp_score1 = partial_sims(comp_i);
        comp_score = comp_scores(comp_i);
    %         if comp_score > prctile(comp_scores,90) | comp_score < prctile(comp_scores,10)
        rectangle('Position',coords(:,comp_i)','EdgeColor', 'red');
        text('Position',coords(1:2 ,comp_i)' + [0 6],'string',sprintf('%.2f', comp_score1), 'Color', 'red')
        text('Position',coords(1:2 ,comp_i)' + [12 6],'string',sprintf('%.2f', color_partial_sims(comp_i)), 'Color', 'cyan')
        text('Position',coords(1:2 ,comp_i)' + [0 12],'string',sprintf('%.2f', overlaps(i, comp_i)), 'Color', 'green')
        text('Position',coords(1:2 ,comp_i)' + [12 12],'string',sprintf('%.2f', comp_score), 'Color', 'yellow')
    %         end
    end
    hold off
end

