function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);

% save distance and label for one sample for all training samples
distances = zeros(size(XTrain,1),2);

for i = 1:length(X)
    D = pdist2(X(i,:), XTrain, 'euclidean');
    distances(:,1) = D;
    distances(:,2) = LTrain;
    distances = sortrows(distances, 1);
    neighbors = distances(1:k, 2);
    % mode returns most frequent value in a sample
    LPred(i) = mode(neighbors);
end

end

