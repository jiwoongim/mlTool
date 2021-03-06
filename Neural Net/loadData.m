data = [];
testdata = [];
targets = [];
testtargets = [];

%Loading digit 2
load ../../MNIST/digit2;  
numExamples = floor(size(D,1)/2);
data = [data; D(1:numExamples,:)];
target2 = repmat([1 0 0], numExamples, 1);

%Loading digit 3
load ../../MNISt/digit3;
data = [data; D(1:numExamples,:)];
target3 = repmat([0 1 0 ], numExamples, 1);

%Loading digit 4
load ../../MNISt/digit4;
data = [data; D(1:numExamples,:)];
target4 = repmat([0 0 1], numExamples, 1);

fprintf('Number of train set: %d \n', size(data,1));

%Loading test digit2
load ../../MNIST/test2;
numExamples = floor(size(D,1)/4);
testdata = [testdata; D(1:numExamples,:)];
testtarget2 = repmat([1 0 0], numExamples, 1);

%Loading test digit3
load ../../MNIST/test3;
testdata = [testdata; D(1:numExamples,:)];
testtarget3 = repmat([0 1 0], numExamples, 1);

%Loading test digit4
load ../../MNIST/test4;
testdata = [testdata; D(1:numExamples,:)];
testtarget4 = repmat([0 0 1], numExamples, 1);

targets = [target2; target3; target4];
testtargets = [testtarget2; testtarget3; testtarget4];


%make batch for train
[N,D] = size(data);
T = size(targets,2);
shuffleInd = randperm(N);

%test info
tN = floor(size(testdata,1));
tT = size(testtargets,2);
shuffleIndtest = randperm(tN);
testdata = testdata(shuffleIndtest,:);
testtargets = testtargets(shuffleIndtest, :);


fprintf('Number of test set: %d \n', size(testdata,1));


numBatch = 100;
batchSz = floor(N/numBatch);
batchdata = zeros(batchSz, D, numBatch);
batchTarget = zeros(batchSz, T, numBatch);

for i=1:numBatch,
    batchData(:,:,i) = data(shuffleInd((i-1)*batchSz+1:batchSz*i),:);
    batchTarget(:,:,i) = targets(shuffleInd((i-1)*batchSz+1:batchSz*i),:); 
end;

fprintf('Initializing batch finished\n');












