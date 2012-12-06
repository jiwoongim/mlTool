function [w_rbm hidbias visbias] = rbm_free(batchData, testdata, numHid, epsilon, fmomentum,...
                    w_decay, maxEpoch, start, dataInfo)
%%%
% batchdata - data
% testdata - testdata
% numHid - number of hidden unit
% epsiloon - learning rate
% fmomentum - final momentum
% w_decay - weight decay
% maxEpoch - maximum number of epoch
% start - start training from the beginning 
% dataInfo - information about the data

% hidbias - bias for hidden units
% viisbias - bias for visible units
% w_rbm - weight frmo input to hidden layer%%% 
%%%

%init
tiny = exp(-100);
D = size(batchData,2);
momentum = 0.5;
gw_rbm = 0;
gbias_hid = 0; 
gbias_vis = 0;
numCD = 1;

if start,
    %init random weights from normal distribution
    w_rbm = rand(D,numHid) * 0.1;
    hidbias = zeros(1,numHid);%0.5.* ones(1,numHid);
    visbias = zeros(1,D);%0.5.* ones(1,numHid);
else,
    load W_hidout; 
    load errorList;
    eps_bias = zeros(1,dataInfo.T) * 0.1;
end


for i=1:maxEpoch,

    err = 0; %Reconstruction error        
    for jbatch=1:dataInfo.numBatch,
        data = reshape(batchData(:,:,jbatch), dataInfo.batchSz, D);
        vis = data ;%> rand(size(data));
       
        %Contrastive and divergence 1times in the beginning and 5 times later stage
        [vis dEdw_rbm, dbiasHid, dbiasVis] = contrastive_divergence(numCD, w_rbm, hidbias, visbias, dataInfo.batchSz, vis);
  
        %Converting momentum to 0.9, and setting to CD-5
        if (i > 2 * maxEpoch/3),
            fmomentum = momentum;
            numCD = 5;
            epsilon = epsilon *0.5;
        elseif (i > maxEpoch/3),
            numCD = 3;
            epsilon = epsilon *0.5;
        end

        %Updating weights;
        gw_rbm = (momentum.*gw_rbm + epsilon.*(dEdw_rbm-w_decay*w_rbm));
        gbias_hid = (momentum.*gbias_hid + epsilon.*dbiasHid);
        gbias_vis = (momentum.*gbias_vis + epsilon.*dbiasVis);
       
        w_rbm = w_rbm + gw_rbm;
        hidbias = hidbias + gbias_hid;
        visbias = visbias + gbias_vis;
         
        %Accumulating reconstruction error 
        err = err+ sum(sum( (data - vis).*(data-vis))); %reconstruction error compute
    end
    
    %Error measure
    free_eng = compute_free_energy(data, w_rbm, hidbias, visbias)/dataInfo.batchSz;
    %testvis = testdata > rand(size(testdata));
    free_eng_test = compute_free_energy(testdata, w_rbm, hidbias, visbias)/size(testdata,1);
    diff_free_eng = free_eng_test- free_eng;
    fprintf('%d, Free Eng %f, Free Eng Test %f Avg_Diff %f error %f\n', i, ...
                                free_eng, free_eng_test, diff_free_eng, err);
end

%Show image
colormap(gray);
k = sqrt(D);
figure(1);
displayFaces(reshape(data, dataInfo.batchSz,k,k));
figure(2);
displayFaces(reshape(vis, dataInfo.batchSz,k,k));

end

function free_eng = compute_free_energy(vis, w_rbm, hidbias, visbias),
    N = size(vis,1);
    tmp = vis * w_rbm;
    prob = 1./(1+exp(vis * w_rbm + repmat(hidbias, N,1)));
    free_eng = sum(-vis*visbias' - sum(prob.*(vis * w_rbm + repmat(hidbias, N,1)),2) + sum(prob.*log(prob) +(1-prob) .*log(1-prob),2));

    %free_eng = -vis*visbias' - sum(log(1+exp(vis * w_rbm + repmat(hidbias, N,1))),2);
end


