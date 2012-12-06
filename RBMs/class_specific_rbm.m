%Loading digit 2
data2 = []; 
load ../../MNIST/digit2;  
numExamples = floor(size(D,1))/2;
data2 = [data2; D(1:numExamples,:)];
target2 = repmat([1 0 0], numExamples, 1);
[data2, testdata2, target2, testTarget2] = initializeData(data2,target2);
[batchData2, batchTarget2] = makebatch(data2, target2);
%data = reshape(batchData2(:,:,2), size(batchData2,1), size(batchData2,2));
%displayFaces(reshape(data, size(batchData2,1),28,28));


%Loading digit 3
data3 = [];
load ../../MNISt/digit3;
data3 = [data3; D(1:numExamples,:)];
target3 = repmat([0 1 0 ], numExamples, 1);
[batchData3, batchTarget3] = makebatch(data3, target3);
[data3, testdata3, target3, testTarget3] = initializeData(data3,target3);
[batchData3, batchTarget3] = makebatch(data3, target3);

%Loading digit 4
data4 = [];
load ../../MNISt/digit4;
data4 = [data4; D(1:numExamples,:)];
target4 = repmat([0 0 1], numExamples, 1);
[batchData4, batchTarget4] = makebatch(data4, target4);
[data4, testdata4, target4, testTarget4] = initializeData(data4,target4);
[batchData4, batchTarget4] = makebatch(data4, target4);

dataInfo.numBatch = 50;
dataInfo.batchSz = size(batchData2,1);
dataInfo.D = size(data2, 2);
dataInfo.T = size(target2,2);


%Run RBM
if 0,
    [rbm_w2, obias2, vbias2] = rbm_free(batchData2, testdata2, 300, 0.005, 0.9, 0.0001, 400, 1, dataInfo);
    [rbm_w3, obias3, vbias3] = rbm_free(batchData3, testdata3, 300, 0.01, 0.9, 0.0001, 400, 1, dataInfo);
    [rbm_w4, obias4, vbias4] = rbm_free(batchData4, testdata4, 300, 0.01, 0.9, 0.0001, 400, 1, dataInfo);
end

free_softmax(testdata2, testTarget2);
free_softmax(testdata3, testTarget3);
free_softmax(testdata4, testTarget4);

