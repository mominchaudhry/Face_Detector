%run('../vlfeat-0.9.20/toolbox/vl_setup')
load('my_svm.mat');

scales = [0.1,0.25,0.5,0.75,1,1.25];

temp_bboxes = zeros(0,4);
temp_confidences = zeros(0,1);

cellSize = 8;
dim = 36;
% load and show the image
im = im2single(imread('class.jpg'));
figure(); 
imshow(im);
hold on;
for scale = scales
        image = imresize(im, scale);
        % generate a grid of features across the entire image. you may want to 
        % try generating features more densely (i.e., not in a grid)
        feats = vl_hog(image,cellSize);

        % concatenate the features into 6x6 bins, and classify them (as if they
        % represent 36x36-pixel faces)
        [rows,cols,~] = size(feats);    
        confs = zeros(rows,cols);
        for r=1:rows-8
            for c=1:cols-8

            % create feature vector for the current window and classify it using the SVM model, 
            % take dot product between feature vector and w and add b,
        % store the result in the matrix of confidence scores confs(r,c)
                window = feats(r:r+8, c:c+8,:);
                windowFeat = window(:);
                confs(r,c) = windowFeat'*w + b;

            end
        end
        % get the most confident predictions 
        [~,inds] = sort(confs(:),'descend');
        inds = inds(1:size(confs,2)); % (use a bigger number for better recall)
        for n=1:numel(inds)        
            [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));

            bbox = [ col*cellSize / scale ...  
                     row*cellSize / scale ...
                    (col+cellSize-1)*cellSize / scale ...
                    (row+cellSize-1)*cellSize / scale];
            conf = confs(row,col);     

            if conf > 1.2
                % save         
                temp_bboxes = [temp_bboxes; bbox];
                temp_confidences = [temp_confidences; conf];
            end
        end
end
valid_boxes = false(1,size(temp_confidences,1)); 
for x = 1:size(temp_confidences,1)
        is_valid = true;
        cur = temp_bboxes(x,:);
        for j = find(valid_boxes)
            prev=temp_bboxes(j,:);
            bi=[max(cur(1),prev(1)) ; max(cur(2),prev(2)) ; ...
                min(cur(3),prev(3)) ; min(cur(4),prev(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            if iw>0 && ih>0              
                ua=(cur(3)-cur(1)+1)*(cur(4)-cur(2)+1)+...
                   (prev(3)-prev(1)+1)*(prev(4)-prev(2)+1)-...
                   iw*ih;
                ov=iw*ih/ua;
                center = [(cur(1) + cur(3))/2, (cur(2) + cur(4))/2];
                if ov > 0 || ( center(1) > prev(1) && center(1) < prev(3) && ...
                center(2) > prev(2) && center(2) < prev(4))
            
                    is_valid = false;
                end
            end
        end
        valid_boxes(x) = is_valid;
end

bboxes = zeros(0,4);
confidences = zeros(0,1);
for j = 1:size(valid_boxes,2)
        if valid_boxes(j) == 1
            % plot
            plot_rectangle = [temp_bboxes(j,1), temp_bboxes(j,2); ...
                    temp_bboxes(j,1), temp_bboxes(j,4); ... 
                    temp_bboxes(j,3), temp_bboxes(j,4); ...
                    temp_bboxes(j,3), temp_bboxes(j,2); ...
                    temp_bboxes(j,1), temp_bboxes(j,2)];
            plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
            bboxes = [bboxes; temp_bboxes(j,:)];
            confidences = [confidences; temp_confidences(j,:)];
        end
end