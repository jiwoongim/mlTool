function [dEdW_hidout, dbiasOut, dEdW_inphid, dbiasHid epsilon_hidout, eps_bias] = ...
            backprop(data, resErr, hidX, W_hidout, W_inphid, ...
            old_dEdW, old_bias, epsilon_hidout, eps_bias, eps_wd),

    dEdhidX = (resErr * W_hidout').*((1-hidX).*hidX);


    %gradient of weight hid to output
    dEdW_hidout = (hidX' *resErr) + eps_wd .* W_hidout;% * y').* ((1-hidX).*hidX)'; 
    dbiasOut = sum(resErr,1);

    %gradient of weight input to hid
    dEdW_inphid = (data'*dEdhidX)+ eps_wd * W_inphid;
    dbiasHid = sum(dEdhidX,1);

    %updating learning rate for each connections in hid to out 
    ind1 = find(dEdW_hidout.*old_dEdW > 0);
    epsilon_hidout(ind1) = epsilon_hidout(ind1) + 0.05;
    ind2 = find(dEdW_hidout.*old_dEdW < 0);
    epsilon_hidout(ind2) = epsilon_hidout(ind2) * 0.95;

    ind3 = find(old_bias.*dbiasOut > 0);
    eps_bias(ind3) = eps_bias(ind3) + 0.05;
    ind4 = find(old_bias.*dbiasOut > 0);
    eps_bias(ind4) = eps_bias(ind4) * 0.95;

end 
    



