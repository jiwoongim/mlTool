function [data testdata target testTarget] = initializeData(data, target)

    [N, D] = size(data);
    data = data./255;
    numTest = floor(N/4);
    testdata = data(1:numTest,:); 
    testTarget = target(1:numTest,:);
    data = data(numTest+1:N,:);
    target = target(numTest+1:N,:);

    [N,D] = size(data);
    %variance = (var(data,0,1));
    %data = data./repmat(variance, N,1);
    %meanData = sum(sum(data)) / (N*D);
    %data = data - repmat(meanData, N,D); 

    %[tN, D] = size(testdata);
    %meanTestdata = sum(sum(testdata)) / (tN *D);
    %testdata = testdata - repmat(meanTestdata, tN,D);
    %varianceTestdata = (var(testdata,0,1));
    %testdata = testdata ./ repmat(variance,tN,1);
end
