% %% init net
% clear all
% % feature settings
% addpath ../deep-fbanks
% deep_fbanks_dir = '../deep-fbanks/';
% setup;
% addpath([deep_fbanks_dir 'ColorNaming']); % https://github.com/tinghuiz/dataset_bias/tree/master/feature_extract/color/ColorNaming
% % addpath([deep_fbanks_dir 'mtba']); % http://www.iitk.ac.in/idea/mtba/mtba-win.zip
% % addpath([deep_fbanks_dir 'mixturecode2']); % http://www.lx.it.pt/%7Emtf/mixturecode2.zip
% % addpath(genpath([deep_fbanks_dir 'drtoolbox']));
% path_model_vgg_m = [deep_fbanks_dir 'data/models/imagenet-vgg-m.mat'];
% % path_model_vgg_m = [deep_fbanks_dir 'data/models/imagenet-vgg-verydeep-19.mat']; %fix average image! fix offset!
% 
% dcnn.name = 'dcnnfv' ;
% dcnn.opts.type = 'dcnn';
% dcnn.opts.model = path_model_vgg_m;
% dcnn.opts.layer = 13;
% dcnn.opts.numWords = 64;
% dcnn.opts.encoderType = 'fv';
% 
% opts.useGpu = true;
% opts.gpuId = 1;
% 
% if opts.useGpu
%   gpuDevice(opts.gpuId) ;
% end
% 
% net = load(dcnn.opts.model) ;
% net.layers = net.layers(1:dcnn.opts.layer) ;
% if opts.useGpu
%   net = vl_simplenn_move(net, 'gpu') ;
%   net.useGpu = true ;
% else
%   net = vl_simplenn_move(net, 'cpu') ;
%   net.useGpu = false ;
% end
% 
% %% data settings
% 
% imsize = [300 200];
% jpatch_w = 1/3;
% features_folder = '/media/vips/data/mgodi/fashion_visual_summaries/texture_desc/';
% n_elements = 16+15;
% 
% if exist('dataset.mat', 'file')
%     load('dataset.mat')
% else
%     dataset.root = ['data/FashionStyle14_v1/custom2/'];
%     dataset.pose_root = ['data/pose_results/FashionStyle14_v1/custom2/'];
%     dataset.parsing_root = [ 'data/FashionStyle14_v1_parsing/custom2/'];
%     dataset.sets = {'val', 'test', 'train'};
% %     dataset.sets = {'test'};
%     dataset.classes = dir(fullfile(dataset.root, dataset.sets{1}));
%     dataset.classes = {dataset.classes(3:end).name};
%     for set=dataset.sets
%         set=char(set)
%         dataset.(set).path = fullfile(dataset.root, set);
%         dataset.(set).all = {};
%         dataset.(set).pose_info.all = {};
%         dataset.(set).all_labels = [];
%         dataset.(set).parsing.all = {};
%         count_cl = 0;
%         for cl=dataset.classes
%             count_cl = count_cl + 1;
%             cl = char(cl);
%             tmp = dir(fullfile(dataset.(set).path, cl, '*.*'));
%             tmp = tmp(3:end);
%             dataset.(set).(cl) = cell(size(tmp, 1), 1);
%             dataset.(set).parsing.(cl) = cell(size(tmp, 1), 1);
%             for i=1:size(tmp, 1)
%                 dataset.(set).(cl){i} = fullfile(tmp(i).folder, tmp(i).name);
%                 [filepath,name,ext] = fileparts(dataset.(set).(cl){i});
%                 dataset.(set).parsing.(cl){i} = fullfile(dataset.parsing_root, set, cl, [name '.png']);
% %                 if ~exist(dataset.(set).parsing.(cl){i}, 'file')
% %                     fprintf("ERRORE, NO PARSING\n");
% %                 end
%             end
%             
%             json_path = fullfile(dataset.pose_root, set, cl, 'POSE/alpha-pose-results.json');
%             [dataset.(set).pose_info.(cl), discarded] = decode_pose_json_fn(json_path, dataset.(set).(cl));
%             dataset.(set).(cl) = dataset.(set).(cl)(~discarded);
%             dataset.(set).pose_info.(cl) = dataset.(set).pose_info.(cl)(~discarded);
%             dataset.(set).parsing.(cl) = dataset.(set).parsing.(cl)(~discarded);
%             
%             dataset.(set).all = [dataset.(set).all; dataset.(set).(cl)];
%             dataset.(set).all_labels = [dataset.(set).all_labels; ones(size(dataset.(set).(cl), 1), 1) .* count_cl];
%             dataset.(set).pose_info.all = [dataset.(set).pose_info.all; dataset.(set).pose_info.(cl)];
%             dataset.(set).parsing.all = [dataset.(set).parsing.all; dataset.(set).parsing.(cl)];
% %             feat_vect = compute_img_features_fn(dataset.train.(cl){1},imsize, jpatch_w, dataset.train.pose_info.(cl){1}, net, [])
%         end
%     end
%     save('dataset.mat', 'dataset')
% end
% 
% % % %% extract conv features
% % % n_imgs = 1000;
% % % set = 'train';
% % % %sample randomly from all, maybe do something balanced by class?
% % % enum_i = 0;
% % % %     descrs = zeros(512, n_imgs * n_descr_img);
% % % descrs = cell(n_imgs, 1);
% % % conv_feats_file = 'conv_descrs1000.mat';
% % % % if exist(conv_feats_file, 'file')
% % %     load(conv_feats_file)
% % % % else
% % % %     index_subset = randperm(numel(dataset.(set).all));
% % % %     index_subset = index_subset(1:n_imgs);
% % % %     for enum_i=1:numel(index_subset)
% % % %         i = index_subset(enum_i);
% % % %         tmp_path = dataset.(set).all{i};
% % % %         tmp_pose_info = dataset.(set).pose_info.all{i};
% % % %         tmp_parsing_path = dataset.(set).parsing.all{i};
% % % %         feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, [], tmp_parsing_path);
% % % %         descrs{enum_i} = feat_vect;
% % % %         enum_i
% % % %     end
% % % %     save(conv_feats_file, 'descrs','index_subset','set','-v7.3');
% % % % end
% % % descrs_old = descrs;
% %% extract all conv features
% features_file = fullfile(features_folder, 'conv_features_all.mat');
% features = struct;
% 
% if exist(features_file, 'file')
%     load(features_file)
%     'loaded features'
% else
% 
%     features_file_cp = fullfile(features_folder, 'conv_features_all_cp.mat');
% 
%     if exist(features_file_cp, 'file')
%         fprintf('Trovato checkpoint, loading...\n');
%         checkpoint = load(features_file_cp);
%     end
% 
%     for set=dataset.sets
%         set=char(set);
%         if ~isfield(features, set)
%             tmp_set = cell(size(dataset.(set).all, 1), 1);
%             step = 500;
%             for cursor=0:step:size(dataset.(set).all, 1)+step
%                 if exist('checkpoint', 'var') && (cursor+step) <= checkpoint.i && strcmp(checkpoint.set,  set)
%                     tmp_set(cursor+1:cursor+step) = checkpoint.tmp_set(cursor+1:cursor+step);
%                     continue
%                 end
%                 parfor i=cursor+1:min(step+cursor, size(dataset.(set).all, 1))
%                     i
%                     tmp_path = dataset.(set).all{i};
%                     tmp_pose_info = dataset.(set).pose_info.all{i};
%                     tmp_parsing_path = dataset.(set).parsing.all{i};
%                     tmp_feats = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, [], tmp_parsing_path);
%                     tmp_set{i} = tmp_feats;
%                 end
%                 i = min(step+cursor, size(dataset.(set).all, 1));
%                 fprintf('Salvo checkpoint....\n')
%                 save(features_file_cp, 'set', 'i', 'tmp_set', '-v7.3');
%             end
% 
%             features.(set) = tmp_set;
%             save(features_file, 'features', '-v7.3')
%         end
%     end
% end
%     
%% train gmms
% 
% set = 'train';
% n_imgs = 4000;
% max_descrs_per_img = 16;
% 
% encoders = cell(n_elements, 1);
% if exist('encoder_experimental.mat', 'file')
%     load('encoder_experimental.mat')
%     for el_i=1:n_elements
%         el_i
%         if isfield(encoders{el_i}, 'postpca')
%             encoders{el_i}.encoder.numWords = 8;
%             encoders{el_i}.encoder.projection = 1 ;
%             encoders{el_i}.encoder.projectionCenter = 0 ;
%             encoders{el_i}.encoder.encoderType = dcnn.opts.encoderType;
%             encoders{el_i}.encoder.renormalize = false ;
%             %sample randomly from all, maybe do something balanced by class?
%             enum_i = 0;
%             index_subset = randperm(numel(dataset.(set).all));
%             index_subset = index_subset(1:n_imgs);
%             labels = zeros(0);
%             descrs = zeros(512, 0);
%             for enum_i=index_subset
%                 feats = [];
%                 feats = [feats vl_colsubset(features.(set){enum_i}{el_i}.feats, max_descrs_per_img)];
%     %             labels = [labels ones(1, size(feats, 2)) .* dataset.(set).all_labels(enum_i)];
%                 descrs = [descrs feats];
%             end
% 
%             encoders{el_i}.preprocessing.mean = mean(descrs, 2);
%             encoders{el_i}.preprocessing.std = std(descrs,0,2);
%             descrs = (descrs - encoders{el_i}.preprocessing.mean) ./ encoders{el_i}.preprocessing.std;
% 
%             [coeff,~,latent] = pca(descrs');
%             expl_vars = cumsum(latent) ./ sum(latent);
%             n_dims = find(expl_vars >= 0.95, 1)
%             coeff = coeff(:, 1:n_dims);
%             encoders{el_i}.pca.coeff = coeff;
% 
%             descrs = (descrs' * coeff)';
%             v = var(descrs')' ;
% 
%             [encoders{el_i}.encoder.means, encoders{el_i}.encoder.covariances, encoders{el_i}.encoder.priors] = vl_gmm(descrs, encoders{el_i}.encoder.numWords, 'Initialization', 'kmeans', 'CovarianceBound', double(max(v)*0.0001), 'NumRepetitions', 3);
%             encoders{el_i}.encoder.means = single(encoders{el_i}.encoder.means);
%             encoders{el_i}.encoder.covariances = single(encoders{el_i}.encoder.covariances);
%             encoders{el_i}.encoder.priors = single(encoders{el_i}.encoder.priors);
% 
%             descrs_ifv = zeros(size(descrs, 2), prod(size(encoders{el_i}.encoder.means)) * 2);
%             cursor = 0;
%             clear descrs
%             for enum_i=index_subset
%                 feats = [];
%                 subfeats_subset = randperm(size(features.(set){enum_i}{el_i}.feats, 2));
%                 subfeats_subset = subfeats_subset(1:min([size(features.(set){enum_i}{el_i}.feats, 2) max_descrs_per_img]));
%                 for j=subfeats_subset
%                     tmp_feats = (features.(set){enum_i}{el_i}.feats(:, j) - encoders{el_i}.preprocessing.mean) ./ encoders{el_i}.preprocessing.std;
%                     tmp_feats = tmp_feats' * encoders{el_i}.pca.coeff;
%                     feats = [feats vl_fisher(tmp_feats', encoders{el_i}.encoder.means, encoders{el_i}.encoder.covariances, encoders{el_i}.encoder.priors, 'Improved')];
%                 end
%                 descrs_ifv(cursor+1:cursor+size(feats', 1), :) = feats';
%                 cursor = cursor + size(feats', 1);
%             end
%             descrs_ifv = descrs_ifv(1:cursor, :);
%             [coeff,~,latent] = pca(descrs_ifv);
%             expl_vars = cumsum(latent) ./ sum(latent);
%             n_dims = find(expl_vars >= 0.9, 1)
%             coeff = coeff(:, 1:n_dims);
%             encoders{el_i}.postpca.coeff = coeff;
%             save('encoder_experimental.mat', 'encoders')
%         end
%     end
% end

%% build feat vectors
% set = 'train';
% encoded_features_file = fullfile(features_folder, 'pca_ifv_pca_features_train.mat');
% 
% labels = zeros(0);
% for enum_i=1:size(dataset.(set).all, 1)
%     labels = [labels dataset.(set).all_labels(enum_i)];
% end
% 
% part_dict = [];
% for el_i=1:n_elements
%     part_dict = [part_dict  el_i .* ones(1, size(encoders{el_i}.postpca.coeff, 2))];
% end
% 
% if exist(encoded_features_file, 'file')
%     load(encoded_features_file)
% else
% enc_features = [];
% for enum_i=1:size(dataset.(set).all, 1)
%     enum_i
%     feats = [];
%     for el_i=1:n_elements
%         tmp_feat = features.(set){enum_i}{el_i}.feats;
%         tmp_feat = (tmp_feat - encoders{el_i}.preprocessing.mean) ./ encoders{el_i}.preprocessing.std;
%         tmp_feat = tmp_feat' * encoders{el_i}.pca.coeff;
%         tmp_feat = vl_fisher(tmp_feat', encoders{el_i}.encoder.means, encoders{el_i}.encoder.covariances, encoders{el_i}.encoder.priors, 'Improved') ; %single vs multi??
%         feats = [feats tmp_feat' * encoders{el_i}.postpca.coeff];
%         
%         if isempty(part_dict)
%             tmp_part_dict = [tmp_part_dict  el_i .* ones(1, size(encoders{el_i}.postpca.coeff, 2))];
%         end
%     end
%     
%     if isempty(part_dict)
%         part_dict = tmp_part_dict;
%     end
%     
%     if isempty(enc_features)
%         enc_features = zeros(size(dataset.(set).all, 1), numel(feats));
%     end
%     enc_features(enum_i, :) = feats;
% end
% save(encoded_features_file, 'enc_features', 'labels', 'part_dict','-v7.3')
% end

%% test with cosine knn (full vector)
% n_folds = 5;
% inds = randperm(numel(labels))
% enc_features = enc_features(inds, :);
% labels = labels(inds);
% folds = randi(n_folds,1,numel(labels));
% accs = zeros(n_folds, 1);
% for f=1:n_folds
%     f
%     X_folds = 1:n_folds;
%     X_folds(f) = [];
%     X = enc_features(ismember(folds,X_folds), :);
%     X_labels = labels(ismember(folds,X_folds));
%     Y = enc_features(folds == f, :);
%     Y_labels = labels(folds == f);
%     neighs = knnsearch(X, Y, 'Distance', 'cosine', 'K', 10);
%     neigh_labels = X_labels(neighs);
%     preds = mode(neigh_labels, 2);
%     acc = mean(preds == Y_labels')
%     accs(f) = acc;
% end
% mean(accs(:))

%% test with cosine knn (per part)
sim_mat_allparts = zeros(n_elements, numel(labels), numel(labels));

for el_i=1:n_elements
    enc_features_slice = enc_features(:, part_dict == el_i);
    sims_slice = squareform(pdist(enc_features_slice, 'cosine'));
    sim_mat_allparts(el_i, :, :) = reshape(sims_slice, 1, numel(labels), numel(labels));
end
sim_mat = nanmean(sim_mat_allparts, 1); %distance in realtà

n_folds = 5;
inds = randperm(numel(labels))
enc_features = enc_features(inds, :);
labels = labels(inds);
folds = randi(n_folds,1,numel(labels));
accs = zeros(n_folds, 1);
for f=1
    f
    X_folds = 1:n_folds;
    X_folds(f) = [];
    X_inds = ismember(folds,X_folds);
    Y_inds = folds == f;
    X_labels = labels(X_inds);
    Y_labels = labels(Y_inds);
    fold_dist_mat = sim_mat(Y_inds, X_inds);
    [~, neighs] = sort(fold_dist_mat, 2); %indici all'interno di X_inds
    neighs = neighs(:, 2:10+1);
    neigh_labels = X_labels(neighs);
    preds = mode(neigh_labels, 2);
    acc = mean(preds == Y_labels')
    accs(f) = acc;
end
mean(accs(:))

%% extract color features
% if ~exist('w2c','var')
%     load w2c.mat
% end
% color_features_file = fullfile(features_folder, 'color_features.mat');
% 
% 
% if exist(color_features_file, 'file')
%     load(color_features_file)
% else
%     color_features = struct;
% end
% 
% for set=dataset.sets
%     set=char(set);
%     if ~isfield(color_features, set)
%         tmp_set = zeros(size(dataset.(set).all, 1), 11 * n_elements);
%         tmp_set_overlaps = zeros(size(dataset.(set).all, 1), n_elements);
%         for i=1:size(dataset.(set).all, 1)
%             if mod(i, 100) == 0
%                 i
%             end
%             tmp_path = dataset.(set).all{i};
%             tmp_pose_info = dataset.(set).pose_info.all{i};
%             tmp_parsing_path = dataset.(set).parsing.all{i};
%             tmp_feats = compute_color_features_fn(tmp_path,imsize, jpatch_w, tmp_pose_info, w2c, tmp_parsing_path);
%             tmp_feat_vect = zeros(n_elements, 11);
%             for el=1:n_elements
%                 tmp_feat_vect(el, :) = tmp_feats{el}.feats;
%                 tmp_set_overlaps(i, el) = tmp_feats{el}.overlap_ratio;
%             end
%             tmp_feat_vect = tmp_feat_vect';
%             tmp_set(i, :) = tmp_feat_vect(:);
%         end
%         color_features.(set) = tmp_set;
%         color_features.overlap_scores.(set) = tmp_set_overlaps;
%         save(color_features_file, 'color_features','-v7.3')
%     end
% end

%% compute similarities
% n_joints = 16;
% n_segms = 15;
% n_components = n_joints + n_segms;
% 
% color_lambda = 0.0;
% overlaps = [];
% comp_filter = [];
% 
% set = 'train';
% samples = features;
% color_samples = color_features.(set); color_samples = color_samples(index_subset, :);
% [sim_mat_final, partial_sims_all, color_partial_sims_all] = compute_sim_mat_fn(n_components, part_dict, samples,color_samples,color_lambda,[],0,overlaps);

% %% simple evaluate
% global tmp_custom_distance
% tmp_custom_distance = sim_mat_final;
% 
% distance = 'cosine';
% % distance = @tmpdistfun_fn;
% 
% n_splits = 5;
% assigns = randi(n_splits, 1, size(samples, 1));
% for k=1:2:10
%     k
%     accs = [];
%     for s=1:n_splits
%         tr_folds = 1:5;
%         tr_folds(s) = [];
%         ts_fold = s;
%         trset = find(ismember(assigns, tr_folds));
%         tsset = find(assigns == ts_fold);
%         if ~ischar(distance)
%             preds = labels(mode(knnsearch(trset',tsset', 'Distance', distance, 'K', k), 2));
%         else
%             trfeats = features(trset, :);
%             tsfeats = features(tsset, :);
%             preds = labels(mode(knnsearch(trfeats,tsfeats, 'Distance', distance, 'K', k), 2));
%         end
%         gt = labels(tsset);
%         
%         accs = [accs mean(gt == preds)];
%     end
%     mean(accs)
% end
% 
% %% visualize closest and farthest show where is the similarity
% fake_encoder = struct;
% fake_encoder.imsize = imsize;
% 
% for i=randperm(size(samples, 1))
%     figure
%     [bestval, bestind] = max(sim_mat_final(i, [1:i-1 i+1:end]));
% %     bestind = randi(size(samples, 1));
% %     [worstval, worstind] = min(sim_mat_final(i, [1:i-1 i+1:end]));
%     worstind = randi(size(samples, 1));
%     
%     partial_sims_best = partial_sims_all(:, i, bestind);
%     color_partial_sims_best = color_partial_sims_all(:, i, worstind);
%     partial_sims_worst = partial_sims_all(:, i, worstind);
%     color_partial_sims_worst = color_partial_sims_all(:, i, worstind);
%     
%     visualize_triplet_fn(i,bestind, worstind, partial_sims_best, partial_sims_worst, color_partial_sims_best, color_partial_sims_worst, ...
%                             dataset, imsize, jpatch_w, color_lambda, overlaps, n_components, set, net, fake_encoder);
% 
%     waitforbuttonpress
% %     continue
% %     flag = true;
% %     while flag
% %     [~,~,button]=ginput(1);
% %           switch button
% %               case 3 %left
% %                   print(['similarities_' num2str(i) '_' num2str(bestind)],'-dpng', '-r0');
% %               case 1 %right
% %                 flag = false;
% %           end
% %     end
%     
% end


%% sample conv features and visualize scatter
% n_descr_img = 128;
% descrs = zeros(512, n_imgs * n_descr_img);
% labels = zeros(0);
% for enum_i=1:numel(index_subset)
%     feats = [];
%     for j=1:numel(descrs_old{enum_i})
%         feats = [feats descrs_old{enum_i}{j}.feats];
%     end
%     descrs(:, ((enum_i-1)*n_descr_img)+1:enum_i*n_descr_img) = vl_colsubset(feats, n_descr_img);
%     labels = [labels ones(1, n_descr_img) .* dataset.(set).all_labels(index_subset(enum_i))];
% end
% [coeff,~,latent] = pca(descrs');
% descrs_pca = descrs' * coeff;
% descrs_pca = descrs_pca';
% 
% descrs_std = (descrs - mean(descrs, 2)) ./ std(descrs,0,2);
% [coeff,~,latent] = pca(descrs_std');
% descrs_pca = descrs_std' * coeff;
% descrs_pca = descrs_pca';