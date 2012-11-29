function [W_inphid W_hidout hidbias ybias] = classify(dataInfo, W_inphid, ...
                hidbias,  numHid, epsilon, eps_wd, fmomentum, maxEpoch)
%%%
% N - number of data
% D - dimension of the data
% numBatch - number of batch
% W_inphid - weight from input to hidden units
% W_hidout - weight from hidden units to output units
%%%

T = dataInfo.T; D = dataInfo.D; N =dataInfo.N;
testdata = dataInfo.testing;
testtargets = dataInfo.testtargets;

%initial weights
W_hidout = randn(numHid,T)*0.1;
eps_hidout = randn(numHid,T);
eps_bias = randn(1,T);
ybias = randn(1,T)*0.1;%0.5.*ones(1,T);

%init
tiny = exp(-100);
classErrL = zeros(maxEpoch,1);
crossEntL = zeros(maxEpoch,1);
classErrTot =[];
crossEntTot =[];
classErrTestTot =[];
crossEntTestTot =[];

gW_hidout = 0;  gW_inphid = 0;
gbias_out = 0;  gbias_hid = 0; 

for i=1:maxEpoch,
 
    %init for every epoch 
    classErrTest = 0;
    crossEntropyTest = 0;
    classErr = 0;
    crossEntropy = 0;
    resErr = [];
    hidacts = [];
    yacts = [];
    X = [];
        
    for jbatch=1:dataInfo.numBatch,
        %init dropout 
        dropoutProb = randn(numHid,1);
        dropoutInd = find(dropoutProb > 0.5);

        data = reshape(dataInfo.training(:,:,jbatch), dataInfo.batchSz, D);
        target = reshape(dataInfo.target(:,:,jbatch), dataInfo.batchSz, T);

        %forward propagation on Training data

        %hidden layer
        hidSum = data*W_inphid + repmat(hidbias,dataInfo.batchSz,1);
        hidX = 1./(1+ exp(-hidSum));
        hidacts = [hidacts; hidX];

        %output layer 
        unnormalizedY = exp( hidX*W_hidout + repmat(ybias,dataInfo.batchSz,1));
        y = unnormalizedY ./ repmat(sum(unnormalizedY, 1), dataInfo.batchSz,1);
        yacts = [yacts; y];

        %error train measure
        [yMax, I] = max(y');
        [tMax, tI] = max(target');
        classErr = classErr + length(find(I == tI));  %classification error

        crossEntropy = crossEntropy - sum(sum(target.*log(y+tiny)));    
        resErr = [resErr; (y-target)/dataInfo.batchSz];   %residual error   
        X = [X; data];

        %back propagation
        [dEdW_hidout, dbiasOut, dEdW_inphid, dbiasHid, eps_hidout, eps_bias] =... 
        backprop_adpt(X, resErr, hidacts, W_hidout, W_inphid, ...
            gW_hidout, gbias_out, eps_hidout, eps_bias, eps_wd);

        momentum = 0.5;
        if (floor(maxEpoch/2) < i), 
            momentum = fmomentum;
        end

        %No changes for droped out unit.
        dEdW_hidout(dropoutInd,:) = 0;
        dEdW_inphid(:, dropoutInd) = 0;
       
        %weight update
        gW_hidout = (momentum.*gW_hidout - epsilon.*eps_hidout.*dEdW_hidout./dataInfo.batchSz);
        gbias_out = (momentum.*gbias_out - epsilon.*eps_bias.*dbiasOut./dataInfo.batchSz);
        gW_inphid = (momentum.*gW_inphid - epsilon.*dEdW_inphid./dataInfo.batchSz);
        gbias_hid = (momentum.*gbias_hid - epsilon.*dbiasHid./dataInfo.batchSz);
   

        W_hidout = W_hidout + gW_hidout;
        ybias = ybias + gbias_out;
        W_inphid = W_inphid + gW_inphid;
        hidbias = hidbias + gbias_hid;
     
    end  

    %forward propagation onTesting data
    hidSumt = testdata*W_inphid + repmat(hidbias, dataInfo.tN, 1);
    hidXt = 1./(1+exp(-hidSumt));
    unnormalizeYt = exp(+hidXt*W_hidout./2 + repmat(ybias, dataInfo.tN, 1));
    yt = unnormalizeYt ./ repmat(sum(unnormalizeYt, 1), dataInfo.tN, 1);
     
    %error test measure
    [yMaxt, testI] = max(yt');
    [tMaxtest, targettestI] = max(testtargets');

    %classification error
    classErrTest = classErrTest + length(find(testI == targettestI));  
    crossEntropyTest = crossEntropyTest - sum(sum(testtargets.*log(yt+tiny)));    
     

    %storing error measure
    classErrL(i,1) = N - classErr;
    crossEntL(i,1) = crossEntropy;
    classErrLtest(i,1) = dataInfo.tN - classErrTest;
    crossEntLtest(i,1) = crossEntropyTest;

    fprintf('%d class error: %f cross entropy err: %f\n',i, N-classErr, crossEntropy);
    fprintf('%d test class error: %f cross class entropy err: %f\n', i,...
                 dataInfo.tN-classErrTest, crossEntropyTest);
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
legend(sprintf('min vale of %d\n final value of %d', min(crossEntTot), ...
           crossEntTot(i)));

figure(2);
plot(crossEntTestTot);
title(sprintf('Cross Entropy Error on Test Data'));
legend(sprintf('min vale of %d\n final value of %d', min(crossEntTestTot),...
            crossEntTestTot(i)));

figure(3);
plot(classErrTot);
title(sprintf('Classification Error on Training Data'));
legend(sprintf('min value of %d\n final value of %d', min(classErrTot), ...
                    classErrTot(i)));

figure(4);
plot(classErrTestTot);
title(sprintf('Classification Error on Test Data'));
legend(sprintf('min value of %d\n final value of %d', min(classErrTestTot), ...
                    classErrTestTot(i)));


    





