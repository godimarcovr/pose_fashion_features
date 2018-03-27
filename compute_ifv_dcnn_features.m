% clear all
% % feature settings
% addpath ../deep-fbanks
% deep_fbanks_dir = '../deep-fbanks/';
% setup;
% addpath([deep_fbanks_dir 'ColorNaming']); % https://github.com/tinghuiz/dataset_bias/tree/master/feature_extract/color/ColorNaming
% addpath([deep_fbanks_dir 'mtba']); % http://www.iitk.ac.in/idea/mtba/mtba-win.zip
% addpath([deep_fbanks_dir 'mixturecode2']); % http://www.lx.it.pt/%7Emtf/mixturecode2.zip
% addpath(genpath([deep_fbanks_dir 'drtoolbox']));
% % path_model_vgg_m = [deep_fbanks_dir 'data/models/imagenet-vgg-m.mat'];
% path_model_vgg_m = [deep_fbanks_dir 'data/models/imagenet-vgg-verydeep-19.mat']; %fix average image!
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
% features_folder = 'data/';
% n_elements = 16+15;
% 
% if exist('dataset.mat', 'file')
%     load('dataset.mat')
% else
%     dataset.root = ['data/FashionStyle14_v1/custom2/'];
%     dataset.pose_root = ['data/pose_results/FashionStyle14_v1/custom2/'];
%     dataset.parsing_root = [ 'data/FashionStyle14_v1_parsing/custom2/'];
%     dataset.sets = {'train', 'test', 'val'};
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
% %% fit GMM 
% if exist('encoder.mat', 'file')
%     load('encoder.mat')
% else
%     n_imgs = 500;
%     n_descr_img = 128;
%     encoder.numWords = dcnn.opts.numWords;
%     encoder.projection = 1 ;
%     encoder.projectionCenter = 0 ;
%     encoder.encoderType = dcnn.opts.encoderType;
%     encoder.renormalize = false ;
%     set = 'train';
%     %sample randomly from all, maybe do something balanced by class?
%     enum_i = 0;
%     index_subset = randperm(numel(dataset.(set).all));
%     index_subset = index_subset(1:n_imgs);
% %     descrs = zeros(512, n_imgs * n_descr_img);
%     descrs = cell(n_imgs, 1);
%     parfor enum_i=1:numel(index_subset)
%         i = index_subset(enum_i);
%         tmp_path = dataset.(set).all{i};
%         tmp_pose_info = dataset.(set).pose_info.all{i};
%         tmp_parsing_path = dataset.(set).parsing.all{i};
%         feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, [], tmp_parsing_path);
%         feats = [];
%         for j=1:numel(feat_vect)
%             feats = [feats feat_vect{j}.feats];
%         end
%         if size(feats, 2) < n_descr_img
%             feats = [feats feats(:, 1:n_descr_img-size(feats, 2))];
%         end
% %         enum_i = enum_i + 1;
% %         descrs(:, ((enum_i-1)*n_descr_img)+1:enum_i*n_descr_img) = vl_colsubset(feats, n_descr_img);
%         descrs{enum_i} = vl_colsubset(feats, n_descr_img);
%         enum_i
%     end
%     descrs_old = descrs;
%     descrs = zeros(512, n_imgs * n_descr_img);
%     for enum_i=1:numel(index_subset)
%         descrs(:, ((enum_i-1)*n_descr_img)+1:enum_i*n_descr_img) = descrs_old{enum_i};
%     end
%     clear descrs_old
%     v = var(descrs')' ;
%     [encoder.means, encoder.covariances, encoder.priors] = vl_gmm(descrs, encoder.numWords, 'verbose', 'Initialization', 'kmeans', 'CovarianceBound', double(max(v)*0.0001), 'NumRepetitions', 1);
%     encoder.means = single(encoder.means);
%     encoder.covariances = single(encoder.covariances);
%     encoder.priors = single(encoder.priors);
% %     feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, encoder);
%     save('encoder.mat', 'encoder')
% end
% 
% %% compute PCA
% features_file = fullfile(features_folder, 'features_for_pca.mat');
% n_pca_samples = 1000;
% set='test';
% 
% if exist(features_file, 'file')
%     load(features_file)
% else
%     tmp_set = cell(n_pca_samples, 1);
%     index_subset = randperm(numel(dataset.(set).all));
%     index_subset = index_subset(1:n_pca_samples);
% %     enum_i = 0;
%     %find(~cellfun(@isempty, features.(set)))
%     parfor enum_i=1:numel(index_subset)
% %         enum_i = enum_i + 1
%         enum_i
%         i = index_subset(enum_i);
%         if isempty(tmp_set{enum_i})
%             tmp_path = dataset.(set).all{i};
%             tmp_pose_info = dataset.(set).pose_info.all{i};
%             tmp_parsing_path = dataset.(set).parsing.all{i};
%             tmp_set{enum_i} = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, encoder, tmp_parsing_path);
%         end
%     end
%     features.(set) = tmp_set;
%     clear tmp_set
%     save(features_file, 'features', 'index_subset', '-v7.3')
% end
% 
% pca_samples = find(~cellfun(@isempty, features.(set)));
% pca_samples = pca_samples(randperm(numel(pca_samples)));
% pca_samples = pca_samples(1:n_pca_samples)';
% 
% for el=1:n_elements
%     pca_info_file = fullfile(features_folder, ['pca' sprintf('%02d', el) '.mat']);
%     if ~exist(pca_info_file, 'file')
%         element_feats = zeros(n_pca_samples,numel(encoder.means) * 2, 'single');
%         enum_sample = 0;
%         for sample_ind=pca_samples
%             tmp_feat_info = features.(set){sample_ind}{el};
%             if tmp_feat_info.element ~= el
%                 fprintf('No element at %d \n', sample_ind)
%             else
%                 enum_sample = enum_sample + 1;
%                 element_feats(enum_sample, :) = tmp_feat_info.feats;
%             end
%         end
%         element_feats = element_feats(1:enum_sample, :);
%         element_means = mean(element_feats);
%         element_feats = bsxfun(@minus,element_feats,element_means);
%         [coeff,~,latent] = pca(element_feats);
%         pca_info.coeff = coeff;
%         clear('coeff')
%         pca_info.means = element_means;
%         pca_info.latent = latent;
%         save(pca_info_file, 'pca_info', '-v7.3')
%     end
% end
% 
% clear features
% 
% %% extract features
% features_file = fullfile(features_folder, 'features_reduced.mat');
% features = struct;
% n_dimensions = n_pca_samples - 1;
% 
% pcas = cell(n_elements, 1);
% 
% % figure;
% % hold on
% for el=1:n_elements
%     pca_info_file = fullfile(features_folder, ['pca' sprintf('%02d', el) '.mat']);
%     fprintf('Loading %s \n', pca_info_file)
%     load(pca_info_file)
%     pca_info.coeff = pca_info.coeff(:, 1:n_dimensions);
%     pca_info.latent = pca_info.latent(1:n_dimensions);
% %     plot(cumsum(pca_info.latent) ./ sum(pca_info.latent))
% %     plot(pca_info.latent)
%     pcas{el} = pca_info;
% %     waitforbuttonpress
% end
% 
% 
% if exist(features_file, 'file')
%     load(features_file)
% end
% 
% features_file_cp = fullfile(features_folder, 'features_reduced_cp.mat');
% 
% if exist(features_file_cp, 'file')
%     fprintf('Trovato checkpoint, loading...\n');
%     checkpoint = load(features_file_cp);
% end
% 
% for set=dataset.sets
%     set=char(set);
%     if ~isfield(features, set)
%         tmp_set = zeros(size(dataset.(set).all, 1), n_dimensions * n_elements);
%         step = 100;
%         for cursor=0:step:size(dataset.(set).all, 1)+step
%             if exist('checkpoint', 'var') && (cursor+step) <= checkpoint.i && strcmp(checkpoint.set,  set)
%                 tmp_set(cursor+1:cursor+step, :) = checkpoint.tmp_set(cursor+1:cursor+step, :);
%                 continue
%             end
%             for i=cursor+1:min(step+cursor, size(dataset.(set).all, 1))
%                 i
%                 tmp_path = dataset.(set).all{i};
%                 tmp_pose_info = dataset.(set).pose_info.all{i};
%                 tmp_parsing_path = dataset.(set).parsing.all{i};
%                 tmp_feats = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, encoder, tmp_parsing_path);
%                 tmp_feat_vect = zeros(n_elements, n_dimensions);
%                 for el=1:n_elements
%                     tmp_feat_vect(el, :) = (tmp_feats{el}.feats - pcas{el}.means) * pcas{el}.coeff;
%                 end
%                 tmp_feat_vect = tmp_feat_vect';
%                 tmp_set(i, :) = tmp_feat_vect(:);
%             end
%             fprintf('Salvo checkpoint....\n')
%             save(features_file_cp, 'set', 'i', 'tmp_set');
%         end
%         
%         features.(set) = tmp_set;
%         save(features_file, 'features', '-v7.3')
%     end
% end
% clear pcas
% %% extract color features
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
% clear pca_info
% clear checkpoint
%% clustering
% clear partial_sims_all
% clear color_partial_sims_all
% n_joints = 16;
% n_segms = 15;
% n_components = n_joints + n_segms;
% 
% set = 'test';
% samples = features.(set);
% color_samples = color_features.(set);
% comp_filter = [];
% color_lambda = 0.0;
% symm = true;
% overlaps = color_features.overlap_scores.(set);
% 
% 
% [sim_mat_final, partial_sims_all, color_partial_sims_all] = compute_sim_mat_fn(n_components,n_dimensions, samples,color_samples,color_lambda,comp_filter,symm,overlaps);
% 
% overlaps = color_features.overlap_scores.(set);
% 
% index_cache = [];
% for comp_i=1:n_components
%     index_cache = [index_cache ones(1, n_dimensions) .* comp_i];
% end
% color_index_cache = [];
% for comp_i=1:n_components
%     color_index_cache = [color_index_cache ones(1, 11) .* comp_i];
% end

%% visualize closest and show where is the similarity
% figure
% for i=randperm(size(samples, 1))
%     [bestval, bestind] = max(sim_mat_final(i, [1:i-1 i+1:end]));
% %     bestind = randi(size(samples, 1));
%     [worstval, bestind] = max(sim_mat_final(i, [1:i-1 i+1:end]));
%     
%     partial_sims = partial_sims_all(:, i, bestind);
%     color_partial_sims = color_partial_sims_all(:, i, bestind);
%     
%     visualize_similarities_fn(i,bestind, partial_sims, color_partial_sims, dataset, imsize, jpatch_w, color_lambda, overlaps, n_components, set, net, encoder);
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

%% visualize closest and farthest show where is the similarity
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
%                             dataset, imsize, jpatch_w, color_lambda, overlaps, n_components, set, net, encoder);
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

%% simplicistic evaluation of classification
% 
% for knnk = 1:10
%     good_neighbors_perc = [];
%     for i=1:size(samples, 1)
%         [sims, ind_sims] = sort(sim_mat_final(i, [1:i-1 i+1:end]), 'descend');
%         good_neighbors_perc = [good_neighbors_perc sum(dataset.(set).all_labels(ind_sims(1:knnk)) == dataset.(set).all_labels(i))./knnk];
%     end
%     fprintf('%d neighbors: %.2f same class ratio with std %.2f\n', knnk, mean(good_neighbors_perc), std(good_neighbors_perc));
% end

%% ap clustering
% sim_mat_ap = sim_mat_final - max(sim_mat_final(:));
% [apcls, ~, ~, ~] = apcluster(sim_mat_ap, median(sim_mat_ap(:)) * 3.5);
% numel(unique(apcls))
% accs = [];
% dims = [];
% count_i = 0;
% for i=unique(apcls)'
%     count_i = count_i + 1;
%     inds = find(apcls == i);
%     
%     tmp_labels = dataset.(set).all_labels(inds);
%     dom_lab = mode(tmp_labels);
%     acc = nnz(tmp_labels == dom_lab) / numel(tmp_labels);
%     dims = [dims numel(tmp_labels)];
%     accs = [accs acc];
%       
%     inds = inds(randperm(numel(inds)));
%     inds = inds(1:min(9, numel(inds)));
%     figure
%     for j=1:min(9, numel(inds))
%         ind = inds(j);
%         subplot(3,3,j)
%         imshow(imread(dataset.(set).all{ind}));
%     end
% end
% '**********'
% mean(accs)
% mean(dims)

%% biclustering
% n_joints = 16;
% n_segms = 15;
% n_components = n_joints + n_segms;
% index_cache = [];
% for comp_i=1:n_components
%     index_cache = [index_cache ones(1, n_dimensions) .* comp_i];
% end
% color_index_cache = [];
% for comp_i=1:n_components
%     color_index_cache = [color_index_cache ones(1, 11) .* comp_i];
% end
% 
% set = 'train';
% samples = features.(set);
% color_samples = color_features.(set);
% 
% cats = 1:14;
% samples_inds = false(1, size(samples, 1));
% for c=cats
%     samples_inds = samples_inds | (dataset.(set).all_labels == c)';
% end
% samples_inds = find(samples_inds);
% 
% elements = 1:31;
% column_inds = false(1, size(samples, 2));
% Sz = zeros(size(samples, 2));
% for e=elements
% %     tmp = find(index_cache == e);
% %     tmp = tmp(1:200);
% %     tmp2 = false(1, size(samples, 2));
% %     tmp2(tmp) = true;
% %     column_inds = column_inds | tmp2;
%     column_inds = column_inds | (index_cache == e);
%     Sz(index_cache == e, index_cache == e) = 1.0;
% end
% column_inds = find(column_inds);
% Sz = Sz(column_inds, column_inds);
% 
% samples = samples(samples_inds, column_inds); %-1 1
% % samples(samples<-0.1 | samples > 0.1) = mean(samples(:));
% samples = samples - min(samples(:)); % 0 2
% samples = samples ./ max(samples(:)); % 0 1
% samples = samples .* 10;
% 
% 
% 
% 
% % res = S4B(samples, zeros(size(samples, 1)), Sz,4,'./iterations/', 'thr', 10^-2, 'thrl', 10^-2);
% 
% 
% % resfabia = biclust(samples, 'fabia');
% 
% subset_imgpaths = dataset.(set).all(samples_inds);
% subset_poses = dataset.(set).pose_info.all(samples_inds);
% subset_labels = dataset.(set).all_labels(samples_inds);
% clsfabia = resfabia.Clust;
% 
% class_distr = zeros(numel(clsfabia), numel(cats));
% 
% for i=1:numel(clsfabia)
%     tmp_rows = clsfabia(i).rows;
%     tmp_cols = clsfabia(i).cols;
%     bcl_imgpaths = subset_imgpaths(tmp_rows);
%     bcl_poses = subset_poses(tmp_rows);
%     bcl_labels = subset_labels(tmp_rows);
%     
%     tmp_joint_occs = index_cache(tmp_cols);
%     tmp_joints = unique(tmp_joint_occs);
%     
%     tmp_labels = unique(bcl_labels)';
%     [tmp_counts, tmp_classes] = histc(bcl_labels, tmp_labels);
%     
%     class_distr(i, tmp_labels) = tmp_counts;
%     
%     tmp_joints; 
%     figure
% %     for j=1:min(4, numel(tmp_rows))
%     for j=randperm(numel(tmp_rows))
%         
% %         subplot(2,2,j)
%         ind = tmp_rows(j); %sul subset delle classi scelto
%         img = imread(bcl_imgpaths{j});
%         pose_info = bcl_poses{j};
%         
%         sc_x = size(img, 2) / imsize(2);
%         sc_y = size(img, 1) / imsize(1);
%         img = imresize(img, imsize);
%         imshow(img);
%         title(num2str(bcl_labels(j)));
%         jpatch_size = jpatch_w * (pose_info.width / sc_x);
%         
%         hold on
%         coords = [pose_info.coords floor((pose_info.segments(1:2, :) + pose_info.segments(3:4, :)) ./ 2)];
%         coords = [coords./[sc_x; sc_y]; ones(2, n_components).*jpatch_size];
%         coords(1:2, :) = coords(1:2, :) - floor(jpatch_size/2);
%         coords(1:2, coords<0) = 0;
%         
%         comp_scores = [];
%         for k=1:numel(tmp_joints)
%             comp_i = tmp_joints(k);
%             comp_score = sum(tmp_joint_occs == comp_i) / n_dimensions;
% %             rectangle('Position',coords(:,comp_i)','EdgeColor', 'red');
% %             text('Position',coords(1:2 ,comp_i)' + [0 6],'string',sprintf('%.2f', comp_score), 'Color', 'red')
%             comp_scores = [comp_scores comp_score];
%         end
%         [~,~,button]=ginput(1);
%           switch button
%               case 28 %left
%                   break
%               case 29 %right
%                 continue
%            end
%     end
%     close all
%     'min comp score'
%     min(comp_scores)
%     'max comp score'
%     max(comp_scores)
%     '******'
%     
%     
% end
% class_distr
% figure
% imagesc(class_distr ./ sum(class_distr, 2))

%% extract conv features
% n_imgs = 1000;
% n_descr_img = 128;
% encoder.numWords = dcnn.opts.numWords;
% encoder.projection = 1 ;
% encoder.projectionCenter = 0 ;
% encoder.encoderType = dcnn.opts.encoderType;
% encoder.renormalize = false ;
% set = 'train';
% %sample randomly from all, maybe do something balanced by class?
% enum_i = 0;
% index_subset = randperm(numel(dataset.(set).all));
% index_subset = index_subset(1:n_imgs);
% %     descrs = zeros(512, n_imgs * n_descr_img);
% descrs = cell(n_imgs, 1);
% parfor enum_i=1:numel(index_subset)
%     i = index_subset(enum_i);
%     tmp_path = dataset.(set).all{i};
%     tmp_pose_info = dataset.(set).pose_info.all{i};
%     tmp_parsing_path = dataset.(set).parsing.all{i};
%     feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, [], tmp_parsing_path);
%     descrs{enum_i} = feat_vect;
%     enum_i
% end
% save('conv_descrs1000.mat', 'descrs','index_subset','set','-v7.3');
%%
clear all
load('conv_descrs1000.mat');
load('dataset.mat');
n_imgs = 1000;
n_descr_img = 128;

descrs_old = descrs;
descrs = zeros(512, 0);
labels = zeros(0);
for enum_i=1:numel(index_subset)
feats = [];
for j=[13 14]
    if any(ismember(dataset.(set).all_labels(index_subset(enum_i)), 1:14))
        feats = [feats descrs_old{enum_i}{j}.feats];
        labels = [labels ones(1, size(descrs_old{enum_i}{j}.feats, 2)) .* dataset.(set).all_labels(index_subset(enum_i))];
    end
end
descrs = [descrs feats];
end
% descrs = descrs';
% [descrs_white, whiteningW, mu_descrs] = prewhiten(descrs);
% 
% [mapped_descrs_white, mapping_descrs_white] = compute_mapping(descrs_white, 'KernelPCA' , 3, 'linear');
% figure
% hold on
% scatter3(mapped_descrs_white(:, 1), mapped_descrs_white(:, 2), mapped_descrs_white(:, 3), 36, labels);
% cmap = jet(2);    %or build a custom color map
% colormap(cmap);
% hold off
% 
% [mapped_descrs_white, mapping_descrs_white] = compute_mapping(descrs_white, 'KernelPCA' , 3, 'gauss',1);
% figure
% hold on
% scatter3(mapped_descrs_white(:, 1), mapped_descrs_white(:, 2), mapped_descrs_white(:, 3), 36, labels);
% cmap = jet(2);    %or build a custom color map
% colormap(cmap);
% hold off

% descrs_std = (descrs - mean(descrs, 2)) ./ std(descrs,0,2);
% [coeff,~,latent] = pca(descrs_std');
% descrs_pca = descrs_std' * coeff;
% descrs_pca = descrs_pca';


descrs_old = descrs;
descrs = zeros(512, n_imgs * n_descr_img);
labels = zeros(0);
for enum_i=1:numel(index_subset)
    feats = [];
    for j=1:numel(descrs_old{enum_i})
        feats = [feats descrs_old{enum_i}{j}.feats];
    end
    descrs(:, ((enum_i-1)*n_descr_img)+1:enum_i*n_descr_img) = vl_colsubset(feats, n_descr_img);
    labels = [labels ones(1, n_descr_img) .* dataset.(set).all_labels(index_subset(enum_i))];
end
clear descrs_old
[descrs_white, whiteningW, mu_descrs] = prewhiten(descrs');
descrs = descrs_white';
% v = var(descrs')' ;





ks = [5 6 7 64];
nK = numel(ks);
Sigma = {'full'};
nSigma = numel(Sigma);
SharedCovariance = {false};
SCtext = {'false'};
nSC = numel(SharedCovariance);
RegularizationValue = 0.01;
% Preallocation
gm = cell(nK,nSigma,nSC);
aic = zeros(nK,nSigma,nSC);
bic = zeros(nK,nSigma,nSC);
converged = false(nK,nSigma,nSC);
options = statset('MaxIter',10000);

inds = randperm(size(descrs, 2));
inds = inds(1:80000);


for m = 1:nSC
    for j = 1:nSigma
        for i = 1:nK
            fprintf('\n*******\n %d ; %s ; %d',SharedCovariance{m}, Sigma{j},ks(i));
            gm{i,j,m} = fitgmdist(descrs(:, inds)',ks(i),...
                'CovarianceType',Sigma{j},...
                'SharedCovariance',SharedCovariance{m},...
                'RegularizationValue',RegularizationValue,...
                'Options', options);
            aic(i,j,m) = gm{i,j,m}.AIC;
            bic(i,j,m) = gm{i,j,m}.BIC;
            converged(i,j,m) = gm{i,j,m}.Converged;
        end
    end
end

figure;
bar(reshape(aic,nK,nSigma*nSC));
title('AIC For Various $k$ and $\Sigma$ Choices','Interpreter','latex');
xlabel('$k$','Interpreter','Latex');
ylabel('AIC');
legend({'Diagonal-shared','Full-shared','Diagonal-unshared',...
    'Full-unshared'});

figure;
bar(reshape(bic,nK,nSigma*nSC));
title('BIC For Various $k$ and $\Sigma$ Choices','Interpreter','latex');
xlabel('$c$','Interpreter','Latex');
ylabel('BIC');
legend({'Diagonal-shared','Full-shared','Diagonal-unshared',...
    'Full-unshared'});

% [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(descrs,1,64,0,1e-4,0);
[encoder.means, encoder.covariances, encoder.priors] = vl_gmm(descrs, encoder.numWords, 'verbose', 'Initialization', 'kmeans', 'CovarianceBound', double(max(v)*0.0001), 'NumRepetitions', 1);
encoder.means = single(encoder.means);
encoder.covariances = single(encoder.covariances);
encoder.priors = single(encoder.priors);
%     feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, encoder);



% descrs_old = descrs;
% descrs = zeros(512, n_imgs * n_descr_img);
% for enum_i=1:numel(index_subset)
%     descrs(:, ((enum_i-1)*n_descr_img)+1:enum_i*n_descr_img) = descrs_old{enum_i};
% end
% clear descrs_old
% v = var(descrs')' ;
% [encoder.means, encoder.covariances, encoder.priors] = vl_gmm(descrs, encoder.numWords, 'verbose', 'Initialization', 'kmeans', 'CovarianceBound', double(max(v)*0.0001), 'NumRepetitions', 1);
% encoder.means = single(encoder.means);
% encoder.covariances = single(encoder.covariances);
% encoder.priors = single(encoder.priors);
% %     feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, encoder);
% save('encoder.mat', 'encoder')


%% 








% figure
% for i=randperm(size(samples, 1))
%     [sims, ind_sims] = sort(sim_mat_final(i, :), 'descend');
%     subplot(1,2,1)
%     img = imread(dataset.(set).all{i});
%     pose_info = dataset.(set).pose_info.all{i};
%     sc_x = size(img, 2) / imsize(2);
%     sc_y = size(img, 1) / imsize(1);
%     img = imresize(img, imsize);
%     imshow(img);
%     title(num2str(dataset.(set).all_labels(i)));
%     jpatch_size = jpatch_w * (pose_info.width / sc_x);
%     title(num2str(dataset.(set).all_labels(i)));
%     hold on
%     coords = pose_info.coords;
%     coords = coords(:, comp_filter);
%     coords = [coords./[sc_x; sc_y]; ones(2, 1).*jpatch_size];
%     coords(1:2) = coords(1:2) - floor(jpatch_size/2);
%     coords(1:2, coords<0) = 0;
%     
% 
%     rectangle('Position',coords(:)','EdgeColor', 'red');
%     hold off
%     
%     for j=1:3
%         
%         subplot(1,2,2)
%         bestind = ind_sims(j+1);
%         img = imread(dataset.(set).all{bestind});
%         pose_info = dataset.(set).pose_info.all{bestind};
%         sc_x = size(img, 2) / imsize(2);
%         sc_y = size(img, 1) / imsize(1);
%         img = imresize(img, imsize);
%         imshow(img);
%         title(num2str(dataset.(set).all_labels(bestind)));
%         jpatch_size = jpatch_w * (pose_info.width / sc_x);
%         
%         hold on
%         coords = pose_info.coords
%         coords = coords(:, comp_filter);
%         coords = [coords./[sc_x; sc_y]; ones(2, 1).*jpatch_size];
%         coords(1:2) = coords(1:2) - floor(jpatch_size/2);
%         coords(1:2, coords<0) = 0;
% 
%         comp_score1 = sims(j+1);
%         color_sim = 1 - pdist([color_samples(i, :); color_samples(bestind, :)], 'cosine');
% 
%         rectangle('Position',coords(:)','EdgeColor', 'red');
%         text('Position',coords(1:2)' + [0 6],'string',sprintf('%.2f', comp_score1), 'Color', 'red')
%         text('Position',coords(1:2 )' + [0 18],'string',sprintf('%.2f', color_sim), 'Color', 'cyan')
%         hold off
%         
%         waitforbuttonpress
%         print(['similarities_' num2str(i) '_' num2str(bestind)],'-dpng', '-r0');
%     end
%     
% end

% figure
% for i=randperm(size(samples, 1))
%     subplot(1,2,1)
%     imshow(imread(dataset.(set).all{i}));
%     title(num2str(dataset.(set).all_labels(i)));
%     subplot(1,2,2)
%     [bestval, bestind] = max(sim_mat_final(i, [1:i-1 i+1:end]));
%     imshow(imread(dataset.(set).all{bestind}));
%     title(num2str(dataset.(set).all_labels(bestind)));
%     
%     %*****
%     subplot(1,2,1)
%     
%     partial_sims = zeros(n_components, 1);
%     color_partial_sims = zeros(n_components, 1);
%     feat1 = samples(i, :);
%     feat2 = samples(bestind, :);
%     color_feat1 = color_samples(i, :);
%     color_feat2 = color_samples(bestind, :);
%     for comp_i=1:n_components
%         partial_sims(comp_i) = (feat1(index_cache == comp_i) * feat2(index_cache == comp_i)');% / prod(vecnorm([feat1(index_cache == comp_i); feat2(index_cache == comp_i)],2,2));
%         color_partial_sims(comp_i) = 1.0 - pdist([color_feat1(color_index_cache == comp_i); color_feat2(color_index_cache == comp_i)],'cosine');
%     end
%     partial_sims = partial_sims ./ prod(vecnorm([feat1; feat2],2,2));
% %     partial_sims = (partial_sims ./ 2) + (0.5);
%     partial_sims = partial_sims ./ max(abs(partial_sims));
%     
%     img = imread(dataset.(set).all{i});
%     pose_info = dataset.(set).pose_info.all{i};
%     sc_x = size(img, 2) / imsize(2);
%     sc_y = size(img, 1) / imsize(1);
%     img = imresize(img, imsize);
%     imshow(img);
%     title(num2str(dataset.(set).all_labels(i)));
%     jpatch_size = jpatch_w * (pose_info.width / sc_x);
%     
%     hold on
%     coords = [pose_info.coords floor((pose_info.segments(1:2, :) + pose_info.segments(3:4, :)) ./ 2)];
%     coords = [coords./[sc_x; sc_y]; ones(2, n_components).*jpatch_size];
%     coords(1:2, :) = coords(1:2, :) - floor(jpatch_size/2);
%     coords(1:2, coords<0) = 0;
% %     coords(3, coords(1, :) + coords(3, :) > imsize(2)) = imsize(2);
%     
%     for comp_i=1:n_components
%         comp_score1 = partial_sims(comp_i);% / sum(abs(partial_sims));
%         comp_score = comp_score1 * (1 - color_lambda)  + color_lambda * color_partial_sims(comp_i);
%         if abs(comp_score) > 0.5
%             rectangle('Position',coords(:,comp_i)','EdgeColor', 'red');
%             text('Position',coords(1:2 ,comp_i)' + [0 6],'string',sprintf('%.2f', comp_score1), 'Color', 'red')
%             text('Position',coords(1:2 ,comp_i)' + [0 12],'string',sprintf('%.2f', color_partial_sims(comp_i)), 'Color', 'cyan')
%             text('Position',coords(1:2 ,comp_i)' + [0 18],'string',sprintf('%.2f', overlaps(i, comp_i)), 'Color', 'green')
%         end
%     end
%     
%     %****
%     subplot(1,2,2)
%     img = imread(dataset.(set).all{bestind});
%     pose_info = dataset.(set).pose_info.all{bestind};
%     sc_x = size(img, 2) / imsize(2);
%     sc_y = size(img, 1) / imsize(1);
%     img = imresize(img, imsize);
%     imshow(img);
%     title(num2str(dataset.(set).all_labels(bestind)));
%     jpatch_size = jpatch_w * (pose_info.width / sc_x);
%     
%     hold on
%     coords = [pose_info.coords floor((pose_info.segments(1:2, :) + pose_info.segments(3:4, :)) ./ 2)];
%     coords = [coords./[sc_x; sc_y]; ones(2, n_components).*jpatch_size];
%     coords(1:2, :) = coords(1:2, :) - floor(jpatch_size/2);
%     coords(1:2, coords<0) = 0;
% %     coords(3, coords(1, :) + coords(3, :) > imsize(2)) = imsize(2);
%     
%     for comp_i=1:n_components
%         comp_score1 = partial_sims(comp_i);% / sum(abs(partial_sims));
%         comp_score = comp_score1 * (1 - color_lambda)  + color_lambda * color_partial_sims(comp_i);
%         if abs(comp_score) > 0.5
%             rectangle('Position',coords(:,comp_i)','EdgeColor', 'red');
%             text('Position',coords(1:2 ,comp_i)' + [0 6],'string',sprintf('%.2f', comp_score1), 'Color', 'red')
%             text('Position',coords(1:2 ,comp_i)' + [0 12],'string',sprintf('%.2f', color_partial_sims(comp_i)), 'Color', 'cyan')
%             text('Position',coords(1:2 ,comp_i)' + [0 18],'string',sprintf('%.2f', overlaps(bestind, comp_i)), 'Color', 'green')
%         end
%     end
%     
%     hold off
%     waitforbuttonpress
% %     print(['similarities_' num2str(i)],'-dpng', '-r0');
% end



% 
% 
% %ap clustering
% sim_mat_ap = sim_mat - max(sim_mat(:));
% [apcls, ~, ~, ~] = apcluster(sim_mat_ap, median(sim_mat_ap(:)) * 1.5);
% accs = [];
% dims = [];
% count_i = 0;
% for i=unique(apcls)'
%     count_i = count_i + 1;
%     inds = find(apcls == i);
%     
%     tmp_labels = dataset.(set).all_labels(inds);
%     dom_lab = mode(tmp_labels);
%     acc = nnz(tmp_labels == dom_lab) / numel(tmp_labels);
%     dims = [dims numel(tmp_labels)];
%     accs = [accs acc];
%       
% %     inds = inds(randperm(numel(inds)));
% %     inds = inds(1:min(9, numel(inds)));
% %     figure
% %     for j=1:min(9, numel(inds))
% %         ind = inds(j);
% %         subplot(3,3,j)
% %         imshow(imread(dataset.(set).all{ind}));
% %     end
% end
% '**********'
% mean(accs)
% mean(dims)
% std(dims)

% accs = [];
% dims = [];
% 
% %euclidean Kmeans
% ekm_cls = kmeans(samples,k, 'Distance', 'correlation');
% for i=1:k
%     inds = find(ekm_cls == i);
%     
%     tmp_labels = dataset.(set).all_labels(inds);
%     dom_lab = mode(tmp_labels);
%     acc = nnz(tmp_labels == dom_lab) / numel(tmp_labels);
%     dims = [dims numel(tmp_labels)];
%     accs = [accs acc];
%       
% %     inds = inds(randperm(numel(inds)));
% %     inds = inds(1:min(9, numel(inds)));
% %     figure
% %     for j=1:min(9, numel(inds))
% %         ind = inds(j);
% %         subplot(3,3,j)
% %         imshow(imread(dataset.(set).all{ind}));
% %     end
% end
% '**********'
% mean(accs)
% mean(dims)
% std(dims)









