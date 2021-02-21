% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);

dim = 36;

while n_have < n_want
    % generate random 36x36 crops from the non-face images
    i = randsample(nImages, 1);
    image = rgb2gray(imread(strcat('images_notfaces/', imageList(i,1).name)));
    
    randX = randsample(size(image,2)-dim,1);
    randY = randsample(size(image,1)-dim,1);
    
    crop = image(randY:randY+dim-1, randX:randX+dim-1);
    
    imwrite(crop,strcat(new_imageDir, '/', string(n_have), '.jpg'));
    n_have = n_have+1;
end

mkdir('testing_faces');
mkdir('testing_notfaces')

train_face_dir = 'cropped_training_images_faces';
faces = dir(sprintf('%s/*.jpg',train_face_dir));
n = length(faces);

for i = 1 : 0.2*n
    movefile(strcat(train_face_dir, '/', faces(i,1).name), 'testing_faces');
end

train_noface_dir = 'cropped_training_images_notfaces';
nofaces = dir(sprintf('%s/*.jpg',train_noface_dir));
n2 = length(nofaces);


for i = 1 : 0.2*n2
    movefile(strcat(train_noface_dir, '/', nofaces(i,1).name), 'testing_notfaces');
end
