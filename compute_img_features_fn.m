function feat_vect = compute_img_features_fn(img_path,imsize, jpatch_w, pose_info, net, encoder, parsing_path)
feat_vect = cell(size(pose_info.coords, 2), 1);
img = imread(img_path);
sc_x = size(img, 2) / imsize(2);
sc_y = size(img, 1) / imsize(1);
img = imresize(img, imsize);
jpatch_size = jpatch_w * (pose_info.width / sc_x);

if size(pose_info.coords, 2)<16 || size(pose_info.segments, 2) < 15
    fprintf('Not all coords at %s \n', img_path)
end

showFigures = 0;
if showFigures
    figure
    imshow(img)
    hold on
end

element_count = 0;

for j=1:size(pose_info.coords, 2)
    element_count = element_count + 1;
    x = floor(pose_info.coords(1, j) / sc_x);
    y = floor(pose_info.coords(2, j) / sc_y);
    ymin = max(y-jpatch_size/2, 1);
    ymax = min(y+jpatch_size/2, imsize(1));
    xmin = max(x-jpatch_size/2, 1);
    xmax = min(x+jpatch_size/2, imsize(2));
    regions = create_region_basis_fn(ymin, ymax, xmin, xmax, img, parsing_path, imsize);
    [dcnn_feats, dcnn_locs] = get_dcnn_features(net, img, regions, 'encoder', encoder);
    feat_vect{j} = extract_feats_and_locs(dcnn_feats, dcnn_locs, element_count, regions.overlap_ratio);
    if showFigures
        for k=1:size(feat_vect{j}.locs, 2)
            plot(feat_vect{j}.locs(1, k),feat_vect{j}.locs(2, k),'c*');
        end
        plot(x,y,'co', 'MarkerSize', 12);
    end
end

feat_vect_segm = cell(size(pose_info.segments, 2), 1);
for k=1:size(pose_info.segments, 2)
    element_count = element_count + 1;
    xs = [pose_info.segments(1,k), pose_info.segments(3,k)] ./ sc_x;
    ys = [pose_info.segments(2,k), pose_info.segments(4,k)] ./ sc_y;
    
    ymin = max(min(ys), 1);
    ymax = min(max(ys), imsize(1));
    xmin = max(min(xs), 1);
    xmax = min(max(xs), imsize(2));
    if xmax - xmin < jpatch_size
        xmin = max(xmin - jpatch_size/2, 1);
        xmax = min(xmax + jpatch_size/2, imsize(2));
    end
    if ymax - ymin < jpatch_size
        ymin = max(ymin - jpatch_size/2, 1);
        ymax = min(ymax + jpatch_size/2, imsize(1));
    end
    
    regions = create_region_basis_fn(ymin, ymax, xmin, xmax, img, parsing_path, imsize);
    [dcnn_feats, dcnn_locs] = get_dcnn_features(net, img, regions, 'encoder', encoder);
    feat_vect_segm{k} = extract_feats_and_locs(dcnn_feats, dcnn_locs, element_count, regions.overlap_ratio);
    if showFigures
        for kk=1:size(feat_vect_segm{k}.locs, 2)
            plot(feat_vect_segm{k}.locs(1, kk),feat_vect_segm{k}.locs(2, kk),'r*');
        end
        line(xs, ys, 'Color', 'r');
    end
end

feat_vect = [feat_vect; feat_vect_segm];

end

function regions = create_region_basis_fn(ymin, ymax, xmin, xmax, img, parsing_path, imsize)
    regions.basis = zeros(size(img,1), size(img, 2));
    regions.basis(floor(ymin):floor(ymax), floor(xmin):floor(xmax)) = 1;
    if exist(parsing_path, 'file')
%         se = strel('disk',ceil(jpatch_size * 0.25),4);
        parsing_img = imread(parsing_path);
        parsing_img = imresize(parsing_img, imsize, 'nearest');
        parsing_img = parsing_img(:, :, 1) > 2; %0 BG 1 SKIN 2 HAIR
%         parsing_img = imdilate(parsing_img,se);
        regions.overlap_ratio = sum(sum(double(regions.basis & parsing_img))) / sum(regions.basis(:));
        if sum(sum(double(regions.basis & parsing_img))) > 0 %no overlap
            regions.basis = double(regions.basis & parsing_img); %take whatever
        end
    else
        regions.overlap_ratio = 0;
    end
    regions.basis(1,1) = 2;
    regions.basis(size(img,1), size(img,2)) = 3;
    regions.labels = {1, 2, 3};
end

function feat_vect = extract_feats_and_locs(dcnn_feats, dcnn_locs, element_count, overlap_ratio)
    feat_vect.feats = dcnn_feats{1};
    if iscell(feat_vect.feats)
        feat_vect.feats = feat_vect.feats{1};
    else
        feat_vect.feats = feat_vect.feats(:, 1)';
    end
    feat_vect.locs = dcnn_locs{1}{1};
    feat_vect.element = element_count;
    feat_vect.overlap_ratio = overlap_ratio;
end