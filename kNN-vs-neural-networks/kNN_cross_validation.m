%% kNN cross-validation Script

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training samples

numBins = 3;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

%% Cross-validation
max_k = 25;
k_value_acc = zeros(max_k, 1);
for k = 1:max_k
    avg_acc = zeros(numBins, 1);
    for valid = 1:numBins
        XValid = cat(1, XBins{valid});
        LValid = cat(1,LBins{valid});
        tmp = 1:numBins;
        train = tmp(tmp~=valid);
        XTrain = cat(1,XBins{train});
        LTrain = cat(1,LBins{train});
        
        LPredValid = kNN(XValid , k, XTrain, LTrain);
        cM = calcConfusionMatrix(LPredValid, LValid);
        acc = calcAccuracy(cM);
        avg_acc(valid) = acc;
    end
    k_value_acc(k) = sum(avg_acc) / numBins;
end

[accuracy,optimal_k] = max(k_value_acc)

%% Plot optimal classifications

% Select a subset of the training samples
numBins = 2;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[X, D, L] = loadDataSet( dataSetNr );
[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Divide all data into 2 bins: training and test data
XTrain = XBins{1};
LTrain = LBins{1};
XTest  = XBins{2};
LTest  = LBins{2};

% Classify training data
LPredTrain = kNN(XTrain, optimal_k, XTrain, LTrain);
% Classify test data
LPredTest  = kNN(XTest , optimal_k, XTrain, LTrain);

% Calculate The Confusion Matrix and the Accuracy
cM = calcConfusionMatrix(LPredTest, LTest)
acc = calcAccuracy(cM)

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'kNN', [], k);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end
