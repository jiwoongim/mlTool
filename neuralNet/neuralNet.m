%MNIST data can be downloaded from yann.lecun.com/exdb/mnist

clear;
loadData;

%init
tiny = exp(-100);
epsilon = 0.00001;  %Learning rate
maxEpoch = 100; %maximum epoch
classErrL = zeros(maxEpoch,1);
crossEntL = zeros(maxEpoch,1);
classErrLtest = zeros(maxEpoch,1);
crossEntLtest = zeros(maxEpoch,1);

%Number of hidden units
numHid =  400;
start = 1;

%initial weights
if start,
    %init random weights from normal distribution
    W_inphid = randn(D,numHid);
    W_hidout = randn(numHid,T);
    start = 0;
    classErrTot =[];
    crossEntTot =[];
    classErrTestTot =[];
    crossEntTestTot =[];
else,
    load mnist234_w;
    %load mnist234_b;
    load errorList;
end

hidbias = randn(1,numHid);%0.5.* ones(1,numHid);
ybias = randn(1,T);%0.5.*ones(1,T);

%hidbias = 0.5.* ones(1,numHid);
%ybias = 0.5.*ones(1,T);

for i=1:maxEpoch,
   
    classErrTest = 0;
    crossEntropyTest = 0;
    classErr = 0;
    crossEntropy = 0;
    resErr = [];
    hidacts = [];
    yacts = [];
    X = [];
   
    for jbatch=1:numBatch,
        data = reshape(batchData(:,:,jbatch), batchSz, D);
        target = reshape(batchTarget(:,:,jbatch), batchSz, T);

        %hidden layer
        hidSum = data*W_inphid + repmat(hidbias,batchSz,1);
        hidX = 1./(1+ exp(-hidSum));
        hidacts = [hidacts; hidX];

        %output layer 
        unnormalizedY = exp(- hidX*W_hidout - repmat(ybias,batchSz,1));
        y = unnormalizedY ./ repmat(sum(unnormalizedY, 1), batchSz,1);
        yacts = [yacts; y];

        %error train measure
        [yMax, I] = max(y');
        [tMax, tI] = max(target');
        classErr = classErr + length(find(I == tI));  %classification error

        crossEntropy = crossEntropy - sum(sum(target.*log(y+tiny)));    
        resErr = [resErr; target-y];   %residual error   
        X = [X; data];
    end

    hidSumt = testdata*W_inphid + repmat(hidbias, tN, 1);
    hidXt = 1./(1+exp(-hidSumt));
    unnormalizeYt = exp(-hidXt*W_hidout - repmat(ybias, tN, 1));
    yt = unnormalizeYt ./ repmat(sum(unnormalizeYt, 1), tN, 1);
    
    %error test measure
    [yMaxt, testI] = max(yt');
    [tMaxtest, targettestI] = max(testtargets');
    classErrTest = classErrTest + length(find(testI == targettestI));  %classification error
    crossEntropyTest = crossEntropyTest - sum(sum(testtargets.*log(yt+tiny)));    

    %storing error measure
    classErrL(i,1) = N - classErr;
    crossEntL(i,1) = crossEntropy;
    classErrLtest(i,1) = N - classErrTest;
    crossEntLtest(i,1) = crossEntropyTest;

    %back propagation
    [dEdW_hidout, dbiasOut, dEdW_inphid, dbiasHid] =... 
        backprop(X, resErr, target, yacts, hidacts, W_hidout, W_inphid);


    %weight update
    W_hidout = W_hidout - epsilon.*dEdW_hidout;
    ybias = ybias - epsilon.*dbiasOut;
    
    W_inphid = W_inphid - epsilon.*dEdW_inphid;
    hidbias = hidbias - epsilon.*dbiasHid;

    fprintf('%d class error: %f cross entropy err: %f\n',i, N-classErr, crossEntropy);
    fprintf('%d test class error: %f cross class entropy err: %f\n', i,...
                 tN-classErrTest, crossEntropyTest);

end

classErrTot = [classErrTot; classErrL];
crossEntTot = [crossEntTot; crossEntL];
classErrTestTot = [classErrTestTot; classErrLtest];
crossEntTestTot = [crossEntTestTot; crossEntLtest];


save mnist234_w W_hidout W_inphid;
save mnist234_b ybias hidbias;
save errorList classErrTot crossEntTot classErrTestTot crossEntTestTot;

figure(1);
plot(crossEntTot);
title(sprintf('Cross Entropy Error on Training Data'));

figure(2);
plot(crossEntTestTot);
title(sprintf('Cross Entropy Error on Test Data'));

figure(3);
plot(classErrTot);
title(sprintf('Classification Error on Training Data'));

figure(4);
plot(classErrTestTot);
title(sprintf('Classification Entropy Error on Test Data'));



    






