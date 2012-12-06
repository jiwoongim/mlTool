function [batchData, batchTarget] = makebatch(data, target),

    %Initializing 
    [N, D] = size(data); T = size(target,2);
    shuffleInd = randperm(N);
    numBatch = 50;
    batchSz = floor(N/numBatch);
    batchData = zeros(batchSz, D, numBatch);
    batchTarget = zeros(batchSz, T, numBatch);

    for i=1:numBatch,
        batchData(:,:,i) = data(shuffleInd((i-1)*batchSz+1:batchSz*i),:);
        batchTarget(:,:,i) = target(shuffleInd((i-1)*batchSz+1:batchSz*i),:); 
    end
end


