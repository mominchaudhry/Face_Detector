pos_imageDir = 'testing_faces';
pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
pos_nImages_test = length(pos_imageList);

neg_imageDir = 'testing_notfaces';
neg_imageList = dir(sprintf('%s/*.jpg',neg_imageDir));
neg_nImages_test = length(neg_imageList);

cellSize = 4;
featSize = 31*9^2;

pos_feats_test = zeros(pos_nImages_test,featSize);
for i=1:pos_nImages_test
    im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    pos_feats_test(i,:) = feat(:);
end

neg_feats_test = zeros(neg_nImages_test,featSize);
for i=1:neg_nImages_test
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    neg_feats_test(i,:) = feat(:);
end

save('test_feats.mat','pos_feats_test','neg_feats_test','pos_nImages_test','neg_nImages_test')