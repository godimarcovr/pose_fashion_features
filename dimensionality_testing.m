% clear all
load('conv_descrs1000.mat');
load('dataset.mat');
n_imgs = 1000;
n_descr_img = 128;
descrs_old = descrs;

%% specific parts
% descrs = zeros(512, 0);
% labels = zeros(0);
% for enum_i=1:numel(index_subset)
% feats = [];
% for j=[13 14]
%     if any(ismember(dataset.(set).all_labels(index_subset(enum_i)), 1:14))
%         feats = [feats descrs_old{enum_i}{j}.feats];
%         labels = [labels ones(1, size(descrs_old{enum_i}{j}.feats, 2)) .* dataset.(set).all_labels(index_subset(enum_i))];
%     end
% end
% descrs = [descrs feats];
% end

%% all parts
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
[descrs_white, whiteningW, mu_descrs] = prewhiten(descrs');


%% PCA visualization
% [coeff,scores,latent] = pca(descrs');
% [coeff_white,scores_white,latent_white] = pca(descrs_white);
% figure; plot(cumsum(latent) ./ sum(latent))
% figure; plot(cumsum(latent_white) ./ sum(latent_white))
% figure;
% scatter(scores(labels == 2 | labels == 3, 1),scores(labels == 2 | labels == 3, 2), 36, labels(labels == 2 | labels == 3));
% colormap jet
% figure;
% scatter(scores_white(labels == 2 | labels == 3, 1),scores_white(labels == 2 | labels == 3, 2), 36, labels(labels == 2 | labels == 3));
% colormap jet
% figure;
% scatter3(scores(labels == 2 | labels == 3, 1),scores(labels == 2 | labels == 3, 2), scores(labels == 2 | labels == 3, 3), 36, labels(labels == 2 | labels == 3));
% colormap jet
% figure;
% scatter3(scores_white(labels == 2 | labels == 3, 1),scores_white(labels == 2 | labels == 3, 2), scores_white(labels == 2 | labels == 3, 3), 36, labels(labels == 2 | labels == 3));
% colormap jet

%% fit gmm
descrs = descrs_white';

if exist('test_encoder.mat', 'file')
    load('test_encoder.mat')
else
    encoder.numWords = 5;
    encoder.projection = 1 ;
    encoder.projectionCenter = 0 ;
    encoder.encoderType = 'fv';
    encoder.renormalize = false ;
    set = 'train';
    %sample randomly from all, maybe do something balanced by class?
    v = var(descrs')' ;
    [encoder.means, encoder.covariances, encoder.priors] = vl_gmm(descrs, encoder.numWords, 'verbose', 'Initialization', 'kmeans', 'CovarianceBound', double(max(v)*0.0001), 'NumRepetitions', 3);
    encoder.means = single(encoder.means);
    encoder.covariances = single(encoder.covariances);
    encoder.priors = single(encoder.priors);
%     feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, encoder);
    save('test_encoder.mat', 'encoder', 'whiteningW', 'mu_descrs')
end

%% compute IFV features
if exist('ifv_descrs1000.mat', 'file')
    load('ifv_descrs1000.mat')
else
    set = 'test';
    n_imgs = 1000;
    %sample randomly from all, maybe do something balanced by class?
    enum_i = 0;
    index_subset = randperm(numel(dataset.(set).all));
    index_subset = index_subset(1:n_imgs);
    %     descrs = zeros(512, n_imgs * n_descr_img);
    descrs = cell(n_imgs, 1);
    parfor enum_i=1:numel(index_subset)
        i = index_subset(enum_i);
        tmp_path = dataset.(set).all{i};
        tmp_pose_info = dataset.(set).pose_info.all{i};
        tmp_parsing_path = dataset.(set).parsing.all{i};
        feat_vect = compute_img_features_fn(tmp_path, imsize, jpatch_w, tmp_pose_info, net, [], tmp_parsing_path);
        tmp_feat = cell(numel(feat_vect), 1);
        for j=1:numel(feat_vect)
            tmp_feat{j} = ((feat_vect{j}.feats - mu_descrs')' * whiteningW)';
            tmp_feat{j} = vl_fisher(tmp_feat{j}, encoder.means, encoder.covariances, encoder.priors, 'Improved') ; %di ciascuno separatamente?
        end
        descrs{enum_i} = tmp_feat;
        enum_i
    end
    save('ifv_descrs1000.mat', 'descrs','index_subset','set','-v7.3');
end
%% specific parts
descrs_old = descrs;
descrs = zeros(5120, 0);
labels = zeros(0);
set = 'test'; % stesso di quello sopra!
for enum_i=1:numel(index_subset)
feats = [];
for j=1:31
    if any(ismember(dataset.(set).all_labels(index_subset(enum_i)), 1:14))
        feats = [feats descrs_old{enum_i}{j}];
        labels = [labels dataset.(set).all_labels(index_subset(enum_i))];
    end
end
descrs = [descrs feats];
end

[coeff,scores,latent] = pca(descrs');

descrs = zeros(5120, 0);
labels = zeros(0);
set = 'test'; % stesso di quello sopra!
for enum_i=1:numel(index_subset)
feats = [];
for j=[2 5]
    if any(ismember(dataset.(set).all_labels(index_subset(enum_i)), 2:3))
        feats = [feats descrs_old{enum_i}{j}];
        labels = [labels dataset.(set).all_labels(index_subset(enum_i))];
    end
end
descrs = [descrs feats];
end

descrs = (descrs' * coeff)';

%% PCA visualization



figure; plot(cumsum(latent) ./ sum(latent))
figure;
scatter(scores(labels == 2 | labels == 3, 1),scores(labels == 2 | labels == 3, 2), 36, labels(labels == 2 | labels == 3));
colormap jet
figure;
scatter3(scores(labels == 2 | labels == 3, 1),scores(labels == 2 | labels == 3, 2), scores(labels == 2 | labels == 3, 3), 36, labels(labels == 2 | labels == 3));
colormap jet