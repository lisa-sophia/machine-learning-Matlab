function C = WeakClassifier(T, P, X)
% WEAKCLASSIFIER Classify images using a decision stump.
% Takes a vector X of scalars obtained by applying one Haar feature to all
% training images. Classifies the examples using a decision stump with
% cut-off T and polarity P. Returns a vector C of classifications for all
% examples in X.

% Without using loops, since a loop will be too slow to use
% with a reasonable amount of Haar features and training images.

% One row from xTrain

C = -1 * ones(size(X));
C(P.*X > P.*T) = 1;
