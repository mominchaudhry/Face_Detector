fprintf('\nUsing the svm I trained in part 1 of this assignment,\n');
fprintf('there was many false positives and not many true positives.\n');
fprintf('To fix this, I performed hard negative mining for images of peoples knees\n');
fprintf('and other non face images and re-trained my svm.\n');
fprintf('I also performed more positive image mining for darker and noisy faces\n\n');
fprintf('Also when I implemented non-max suppression and generated features\n');
fprintf('at multiple scales, I got more true positive detections\n\n');
fprintf('Please see detect_class_faces.m for implementation on class.jpg\n');
fprintf('and for the bonus\n');