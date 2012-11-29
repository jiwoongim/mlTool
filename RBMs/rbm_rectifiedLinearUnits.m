%function [w_rbm hidbias visbias] = rbm_gaussianEnergy(batchData, numHid, epsilon, fmomentum, maxEpoch, start, dataInfo)
%%%
% N - number of data
% D - dimension of the data
% numBatch - number of batch
% W_inphid - weight frmo input to hidden layer
%%% 

%init
tiny = exp(-100);
gW_inphid = 0;
gbias_hid = 0; 
gbias_vis = 0;


if start,
    %init random weights from normal distribution
    W_inphid = randn(D,numHid);
    hidbias = randn(1,numHid);%0.5.* ones(1,numHid);
    visbias = randn(1,D);%0.5.* ones(1,numHid);

    %Init Learning Rate on each connection of hiden to output unit
    eps_hidout = randn(numHid,T);
else,
    load mnist234_w;
    %load mnist234_b;
    load errorList;
    eps_bias = randn(1,T);
    
end

for i=1:maxEpoch,

    err = 0;          
    for jbatch=1:numBatch,
        data = reshape(batchData(:,:,jbatch), batchSz, D);
        target = reshape(batchTarget(:,:,jbatch), batchSz, T);
        vis = data;
       
        %Contrastive and divergence 5 times
        for cd=1:5,
        
            %Positive phase  : clamping visible unit
            z = vis * W_inphid + repmat(hidbias, batchSz, 1);
            probHid = 1./(1+exp(-z));
            expHid = vis'*probHid; 

            %Negative phase           
            hid = probHid >rand(size(probHid));
            y = hid * W_inphid' + repmat(visbias, batchSz, 1);
            probVis = max(log(1+exp(y)),0);%+randn(batchSz, D))), 0); 
            expVis = hid'* probVis;
            vis = probVis;%+randn(batchSz, D);

            %Saving <hv>_0
            if (cd == 1),
                origHid = expHid;
                origProbHid = probHid;
                origVis = vis;
            end
        end
        z = vis * W_inphid + repmat(hidbias, batchSz, 1);
        probHid = 1./(1+exp(-z));
        expHid = vis'*probHid; 

        if (i < 10),
            momentum = 0.5;
        end

        %Updating weights;
        dEdW_inphid = (origHid - expHid)/batchSz;
        dbiasHid = (sum(origProbHid) - sum(probHid))/batchSz;
        dbiasVis = (sum(origVis) - sum(vis))/batchSz;
        gW_inphid = (momentum.*gW_inphid + epsilon.*dEdW_inphid);
        gbias_hid = (momentum.*gbias_hid + epsilon.*dbiasHid);
        gbias_vis = (momentum.*gbias_vis + epsilon.*dbiasVis);
       
        W_inphid = W_inphid + gW_inphid;
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
end



                



