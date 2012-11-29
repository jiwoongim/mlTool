function [vis, delta_weight, delta_bias, delta_vis] = contrastive_divergence(numCD, w_rbm, hidbias,...
    visbias, batchSz, vis),
%numCD - number of contrasitive divergence
%rbm_w - weight from input to the output layer
%hibias - bias for hidden unit
%visbias - bias for visible units
%batchSz - size of the batch
%vis - visible units (number of the data as rows and number of visible units as columns)

    for cd=1:numCD,
    
        z = vis * w_rbm + repmat(hidbias, batchSz, 1);
        probHid = 1./(1+exp(-z));
        hid = probHid > rand(size(probHid));
        expHid = vis'*probHid; 

        y = hid * w_rbm' + repmat(visbias, batchSz, 1);
        probVis = 1./(1+exp(-y));
        expVis = hid'* probVis;
        vis = probVis; %>= 0.5;

        %Saving <hv>_0
        if (cd == 1),
            %Storing from Positive phase  
            origHid = expHid;
            origProbHid = probHid;
            origVis = vis;
        end
    end
    
    z = vis * w_rbm + repmat(hidbias, batchSz, 1);
    probHid = 1./(1+exp(-z));
    expHid = vis'*probHid; 
    hid = probHid > rand(size(probHid));

    %Computing delta_W
    delta_weight = (origHid - expHid)/batchSz;
    delta_bias = (sum(origProbHid) - sum(probHid))/batchSz;
    delta_vis = (sum(origVis) - sum(vis))/batchSz;
end



