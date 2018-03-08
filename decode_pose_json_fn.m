function [pose_info, discarded] = decode_pose_json_fn(json_path, image_list)
tmp = read_json_text_fn(json_path);
jsondata = jsondecode(char(tmp{1}));
pose_info = cell(size(image_list, 1), 1);
discarded = false(size(image_list, 1), 1);

segments = [1 2; 2 3; 5 6; 4 5; 3 7; 4 7; 7 8; 8 9; 9 10; 11 12; 12 13; 9 13; 9 14; 14 15; 15 16];

for i=1:size(image_list, 1)
    [filepath,name,ext] = fileparts(image_list{i});
    tmp_ind = strcmp({jsondata.image_id}, [name ext]);
    tmp_ind = find(tmp_ind);
    if numel(tmp_ind) == 0
        discarded(i) = true;
        continue
    end
    if numel(tmp_ind) > 1
        max_area = 0;
        max_ind = 0;
        for p=1:numel(tmp_ind)
            tmp_coords = [jsondata(tmp_ind(p)).keypoints(1:3:end)'; jsondata(tmp_ind(p)).keypoints(2:3:end)'];
            tmp_width = max(tmp_coords(1, :)) - min(tmp_coords(1, :));
            tmp_height = max(tmp_coords(2, :)) - min(tmp_coords(2, :));
            tmp_area = tmp_width * tmp_height;
            if tmp_area > max_area
                max_area = tmp_area;
                max_ind = p;
            end
        end
        tmp_ind = tmp_ind(max_ind);
    end
    
    pose_info{i}.coords = [jsondata(tmp_ind).keypoints(1:3:end)'; jsondata(tmp_ind).keypoints(2:3:end)'];
    pose_info{i}.width = max(pose_info{i}.coords(1, :)) - min(pose_info{i}.coords(1, :));
    pose_info{i}.height = max(pose_info{i}.coords(2, :)) - min(pose_info{i}.coords(2, :));
    pose_info{i}.segments = zeros(4, size(segments, 1));
    count_s = 0;
    for s=segments'
        count_s = count_s + 1;
        s1 = s(1);
        s2 = s(2);
        pose_info{i}.segments(:, count_s) = [pose_info{i}.coords(:, s1); pose_info{i}.coords(:, s2)];
    end
    
    if size(pose_info{i}.coords, 2) < 16
        fprintf('Not all coords at %d \n', i)
    end
end
end

