function [w_xtoh w_ytoh hidbias xbias ybias] = DRBM(batchData, batchTarget, testdata, numHid, epsilon, fmomentum,...
                    w_decay, maxEpoch, dataInfo)
%%%
% batchdata - data
% batchtarget - data label
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
% w_rbm - weight frmo input to hidden layer
%%% 

%init
tiny = exp(-100);
D = size(batchData,2);
C = size(batchTarget,2);
momentum = 0.5;
gw_xtoh = 0;
gw_ytoh = 0;
gbias_hid = 0; 
gbias_vis = 0;
gbias_labels = 0;
numCD = 1;

%init random weights from normal distribution
w_xtoh = randn(D,numHid) * 0.1;
w_ytoh = randn(C,numHid) * 0.1;
hidbias = zeros(1,numHid);%0.5.* ones(1,numHid);
xbias = zeros(1,D);%0.5.* ones(1,numHid);
ybias = zeros(1,C);



for i=1:maxEpoch,

    err = 0; %Reconstruction error        
    for jbatch=1:dataInfo.numBatch,
        data = reshape(batchData(:,:,jbatch), dataInfo.batchSz, D);
        labels = reshape(batchTarget(:,:,jbatch), dataInfo.batchSz, C);
        vis = data;
       
        %Contrastive and divergence 1times in the beginning and 5 times later stage
        [vis dEdw_xtoh, dEdw_ytoh, dbiasHid, dbiasVis dbiasLabels] = contrastiveDivergence(numCD, w_xtoh, w_ytoh, hidbias, xbias, ybias, dataInfo.batchSz, vis, labels);
  
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
        gw_xtoh = (momentum.*gw_xtoh + epsilon.*(dEdw_xtoh-w_decay*w_xtoh));
        gw_ytoh = (momentum.*gw_ytoh + epsilon.*(dEdw_ytoh-w_decay*w_ytoh));
        gbias_hid = (momentum.*gbias_hid + epsilon.*dbiasHid);
        gbias_vis = (momentum.*gbias_vis + epsilon.*dbiasVis);
        gbias_labels = (momentum.*gbias_labels + epsilon.*dbiasLabels);
       
        w_xtoh = w_xtoh + gw_xtoh;
        w_ytoh = w_ytoh + gw_ytoh;
        hidbias = hidbias + gbias_hid;
        xbias = xbias + gbias_vis;
        ybias = ybias + gbias_labels;
         
        %Accumulating reconstruction error 
        err = err+ sum(sum( (data - vis).*(data-vis))); %reconstruction error compute
    end
    
    %Error measure
    free_eng = compute_free_energy(data, w_xtoh, hidbias, xbias)/dataInfo.batchSz;
    %testvis = testdata > rand(size(testdata));
    free_eng_test = compute_free_energy(testdata, w_xtoh, hidbias, xbias)/size(testdata,1);
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

function [vis, delta_w_xtoh, delta_w_ytoh, delta_bias, delta_vis, delta_labels] = contrastiveDivergence(numCD, w_xtoh,...
    w_ytoh, hidbias, xbias, ybias, batchSz, vis, labels),
%numCD - number of contrasitive divergence
%rbm_w - weight from input to the output layer
%hibias - bias for hidden unit
%visbias - bias for visible units
%batchSz - size of the batch
%vis - visible units (number of the data as rows and number of visible units as columns)

    origLabels = labels;
    for cd_i=1:numCD, 
        %Positive phase
        z = labels * w_ytoh + vis * w_xtoh + repmat(hidbias, batchSz, 1);
        probHid = 1./(1+exp(-z));
        hid = probHid > rand(size(probHid));
        expHid = vis'*probHid; 

        %Negative phase
        neg_x = hid * w_xtoh' + repmat(xbias, batchSz, 1);
        prob_x = 1./(1+exp(-neg_x));
        exp_x = hid'* prob_x;
        vis = prob_x > rand(size(prob_x));

        neg_y = hid * w_ytoh' + repmat(ybias, batchSz, 1);
        unnormalized_prob_x = exp(neg_y);
        prob_y = bsxfun(@rdivide, unnormalized_prob_x, sum(unnormalized_prob_x,2)); 
        [tmp, maxInd] = max(prob_y, [], 2); %sampling
        labels = zeros(size(prob_y,1),size(labels,2));
        labels(:, maxInd) = 1; 

        if (cd_i == 1), %Saving <hv>_0
            %Storing from Positive phase  
            origHid = expHid;
            origProbHid = probHid;
            origVis = vis;
        end
    end

    z = labels *w_ytoh + vis * w_xtoh + repmat(hidbias, batchSz, 1);
    probHid = 1./(1+exp(-z));
    expHid = vis'*probHid; 
    hid = probHid > rand(size(probHid));

    %Computing delta_W
    delta_w_xtoh = (origHid - expHid)/batchSz;
    delta_w_ytoh = (origLabels'*origProbHid- labels'*probHid)/batchSz;
    delta_bias = (sum(origProbHid) - sum(probHid))/batchSz;
    delta_vis = (sum(origVis) - sum(vis))/batchSz;
    delta_labels = (sum(origLabels) - sum(labels))/batchSz;
    vis = prob_x;
end





function free_eng = compute_free_energy(vis, w_rbm, hidbias, visbias),
    N = size(vis,1);
    prob = 1./(1+exp(vis * w_rbm + repmat(hidbias, N,1)));
    free_eng = sum(-vis*visbias' - sum(prob.*(vis * w_rbm + repmat(hidbias, N,1)),2) + sum(prob.*log(prob) +(1-prob) .*log(1-prob),2));

    %free_eng = -vis*visbias' - sum(log(1+exp(vis * w_rbm + repmat(hidbias, N,1))),2);
end


