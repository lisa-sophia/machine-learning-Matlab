function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);

for actual = 1:length(LTrue)
    cM(LTrue(actual), LPred(actual)) = cM(LTrue(actual), LPred(actual)) + 1;
end

%tmp = confusionmat(LTrue, LPred)

end
