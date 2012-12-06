function [err] = free_softmax(data, target),
    
    load digit2;
    load digit3;
    load digit4;


    free_eng2 = compute_free_energy(data, rbm_w2, vbias2, obias2);
    free_eng3 = compute_free_energy(data, rbm_w3, vbias3, obias3);
    free_eng4 = compute_free_energy(data, rbm_w4, vbias4, obias4);

    sum_eng = exp(-free_eng2) + exp(-free_eng3) + exp(-free_eng4);
    log_prob2 = exp(-free_eng2)./sum_eng;
    log_prob3 = exp(-free_eng3)./sum_eng;
    log_prob4 = exp(-free_eng4)./sum_eng;

    [log_prob2 log_prob3 log_prob4];
    %[tmp, maxI] = max([log_prob2 log_prob3 log_prob4]');
    [tmp, maxI]= min([free_eng2 free_eng3 free_eng4]');
    [tmp1, targetI] = max(target');

    err = length(find(maxI == targetI));
    tN = size(data,1);
    fprintf('numCorrect: %f Error: %f\n', err, (tN-err)/tN);
end

function free_eng = compute_free_energy(vis, w_rbm, hidbias, visbias),

    N = size(vis,1);
    %set every class bit in turn and find -free energy of the configuration
    
    %hid = vis * w_rbm + repmat(hidbias, N,1);
    %prob = 1./(1+exp(hid));
    
    hid = vis * w_rbm + repmat(visbias, N,1);
    prob = 1./(1+exp(hid));

    %free_eng = -sum(visbias') - sum(prob.*(hid),2) + sum(prob.*log(prob) +(1-prob) .*log(1-prob),2);
    free_eng = -vis*hidbias' - sum(prob.*(hid),2) + sum(prob.*log(prob) +(1-prob) .*log(1-prob),2);
    %free_eng = - sum(prob.*(hid),2) + sum(prob.*log(prob) +(1-prob) .*log(1-prob),2);
    %free_eng = -vis*visbias' - sum(log(1+exp(vis * w_rbm + repmat(hidbias, N,1))),2);
end



