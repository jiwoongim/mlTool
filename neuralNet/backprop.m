function [dEdW_hidout, dbiasOut, dEdW_inphid, dbiasHid] = ...
            backprop(data, resErr, target, y, hidX, W_hidout, W_inphid),

    dEdy = resErr.*(1-y).*y; 
    dEdhidX = (resErr * W_hidout').*((1-hidX).*hidX);

    %gradient of weight hid to output
    dEdW_hidout = (hidX' *resErr);% * y').* ((1-hidX).*hidX)';
    dbiasOut = sum(resErr,1);

    %gradient of weight input to hid
    dEdW_inphid = (data'*dEdhidX);
    dbiasHid = sum(dEdhidX,1);

end 
    



