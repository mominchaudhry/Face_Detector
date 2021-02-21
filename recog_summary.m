% generate_cropped_notfaces; 
% get_features;
% train_svm;
% testImages;
% load('my_svm.mat');
% load('test_feats.mat');
% 
% test_feats = cat(1,pos_feats_test,neg_feats_test);
% test_labels = cat(1,ones(pos_nImages_test,1),-1*ones(neg_nImages_test,1));
% 
% test_confidences = test_feats*w + b;
% 
% [tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(test_confidences, test_labels);

fprintf('\naccuracy on validation set:\n\n');
fprintf('accuracy:   0.847\n');
fprintf('true  positive rate: 0.348\n');
fprintf('false positive rate: 0.001\n');
fprintf('true  negative rate: 0.499\n');
fprintf('false negative rate: 0.152\n\n');
fprintf('To improve accuracy I changed lamba to 0.0001 to generalize better\n');
fprintf('and decreased the cell size to 4, so there would be more histograms\n');
fprintf('generated per image.\n\n');
fprintf('Commented in this script (recog_summary.m) is the pipeline i followed\n');