function log_partition = partition_4_tractableNets(rbm_w),
%This function is for calculating partition function when there are very small number of units in one layer.
%For example, one layer has 256 units and other layer has 10 units. 
%rbm_w  - weight from input layer to output layer

%Init
m = size(rbm_w,1);  % number of output units
M = [0:2^m-1]';    
H = rem(floor(M*pow2(-(m-1):0)),2);     %permutation of all possible state in output layer 
partition = 0;
book_keeping1 = [];


for i=1:size(H,1),
    book_keeping1 = [book_keeping1 rbm_w' * H(i,:)'];
end

%Solving Factorization 
for i=1:size(book_keeping1,2),
    partialSum = 1;
    for j=1:size(book_keeping1,1),
        partialSum = partialSum *(exp(book_keeping1(j,i)) +1);
    end
    partition = partition + partialSum;
end

log_partition = log(partition);
end

