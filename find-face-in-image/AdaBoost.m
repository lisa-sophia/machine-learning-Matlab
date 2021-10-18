%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 2000;
% Number of weak classifiers
nbrWeakClassifiers = 30;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% AdaBoost training

d = ones(size(yTrain)) / nbrTrainImages;

% [feature, threshold, polarity, alpha]
classifiers = zeros(nbrWeakClassifiers, 4);

for classifier = 1 : nbrWeakClassifiers
    classifier % just a print to see training progress
    e_min = inf;
    best_feature = 0;
    best_threshold = 0;
    polarity = 1;
    for k = 1 : nbrHaarFeatures
       e = inf;
       for t = 1 : length(xTrain(k,:))
           p = 1;
           c = WeakClassifier(xTrain(k,t), p, xTrain(k,:)); % ska t vara xTrain(k,t)?
           e = WeakClassifierError(c, d, yTrain);
           if e > 0.5
              p = -1;
              e = 1 - e;
           end
           if e < e_min
            e_min = e;
            best_feature = k;
            best_threshold = xTrain(k,t);
            polarity = p;
           end
       end
    end
    %e_min
    alpha = 0.5 * log((1-e_min)/(e_min+0.0001));
    classifiers(classifier, :) = [best_feature, best_threshold, polarity, alpha];
    h = WeakClassifier(best_threshold, polarity, xTrain(best_feature,:));
    d = d .* exp(-1 * alpha * yTrain .* h);
    d = d/sum(d);
end


%% Evaluate the strong classifier

% classifiers = [feature, threshold, polarity, alpha]
strong_classifier = @(x,n) sign(sum(classifiers(1:n,4).*WeakClassifier(classifiers(1:n,2), classifiers(1:n,3), x(classifiers(1:n,1),:)),1));

%%%

% train and test accuracies using the strong_classifier function
predTrain = strong_classifier(xTrain, nbrWeakClassifiers);
accTrain = 1 - (sum(0.5*abs(predTrain-yTrain)) / nbrTrainImages)

predTest = strong_classifier(xTest, nbrWeakClassifiers);
accTest = 1 - (sum(0.5*abs(predTest-yTest)) / nbrTestImages)

%% Plot the error of the strong classifier as a function of the number of weak classifiers.

accuraciesTest = zeros(nbrWeakClassifiers, 1);
accuraciesTrain = zeros(nbrWeakClassifiers, 1);
for nr = 1:nbrWeakClassifiers
    predTest = strong_classifier(xTest, nr);
    accTest = 1 - (sum(0.5*abs(predTest-yTest)) / nbrTestImages);
    accuraciesTest(nr) = accTest;
    predTrain = strong_classifier(xTrain, nr);
    accTrain = 1 - (sum(0.5*abs(predTrain-yTrain)) / nbrTrainImages);
    accuraciesTrain(nr) = accTrain;
end

[maxAccuracy, optimalNr] = max(accuraciesTest)

figure(4);
plot(1:nbrWeakClassifiers, accuraciesTest);
hold on;
plot(1:nbrWeakClassifiers, accuraciesTrain);
legend('Test','Train');
hold off;

%% Plot some of the misclassified faces and non-faces

missclass = find(yTest ~= predTest);
missclass_faces = missclass(1:5);
missclass_non_faces = missclass((end-5):end);

figure(5);
colormap gray;
for k=1:5
    subplot(2,5,k), imagesc(testImages(:,:,missclass_faces(k)));
    axis image;
    axis off;
end

colormap gray;
for k=1:5
    subplot(2,5,k+5), imagesc(testImages(:,:,missclass_non_faces(k)));
    axis image;
    axis off;
end

%% Plot the choosen Haar-features

figure(6);
colormap gray;
for k = 1:36
    bestFeature = classifiers(k,1);
    subplot(6,6,k),imagesc(haarFeatureMasks(:,:,bestFeature),[-1 2]);
    axis image;
    axis off;
end
