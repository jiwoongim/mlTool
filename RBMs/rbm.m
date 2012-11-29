function [w_rbm hidbias visbias] = rbm(batchData, numHid, epsilon, fmomentum, maxEpoch, start, dataInfo)
%%%
% N - number of data
% D - dimension of the data
% numBatch - number of batch
% w_rbm - weight frmo input to hidden layer
%%% 

%init
tiny = exp(-100);
normalize_data = 0;
D = size(batchData,2);
batchSz = dataInfo.batchSz;
numBatch = dataInfo.numBatch;

if start,
    %init random weights from normal distribution
    w_rbm = randn(D,numHid) * 0.1;
else,
    load mnist234_w;
    %load mnist234_b;
    load errorList;
    eps_bias = rand(1,T) * 0.1;
end

hidbias = randn(1,numHid);%0.5.* ones(1,numHid);
visbias = randn(1,D);%0.5.* ones(1,numHid);

momentum = 0.5;
gw_rbm = 0;
gbias_hid = 0; 
gbias_vis = 0;
numCD = 1;
for i=1:maxEpoch,

    err = 0;       
   
    for jbatch=1:numBatch,
        data = reshape(batchData(:,:,jbatch), batchSz, D);
        vis = data;
       
        %Contrastive and divergence 1times in the beginning and 
        %5 times later stage
        [dEdw_rbm, dbiasHid, dbiasVis] = contrastive_divergence(numCD, w_rbm, hidbias, visbias, batchSz, vis);
              
        %Converting momentum to 0
        if (i > maxEpoch/3),
            fmomentum = momentum;
            numCD = 5;
        end

        %Updating weights;
        gw_rbm = (momentum.*gw_rbm + epsilon.*dEdw_rbm);
        gbias_hid = (momentum.*gbias_hid + epsilon.*dbiasHid);
        gbias_vis = (momentum.*gbias_vis + epsilon.*dbiasVis);
       
        w_rbm = w_rbm + gw_rbm;
        hidbias = hidbias + gbias_hid;
        visbias = visbias + gbias_vis;

        %error compute
        err = err+ sum(sum( (data - vis).*(data-vis)));
    end
    fprintf('%d iteration,  %d batch done, error %f\n', i, jbatch, err);
end

colormap(gray);
k = sqrt(D);
figure(1);
displayFaces(reshape(data, 89,k,k));
figure(2);
displayFaces(reshape(vis, 89,k,k));

%im = reshape(vis, k,k);
%image(im);
save W_hidout w_rbm hidbias visbias


end

                



