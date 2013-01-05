function [] = gaussianProcessDemo(alpha,beta),
% This function demonstrates Gaussian Process with various 
% kernel methods. 
% alpha, beta -  hyperparmeter for kernel function

    %Select uniform samples from the interval
    x = [0:0.005:1];
    n = length(x)
    X = randn(n,1);
   
    %Compute covariance matrix
    Cov = zeros(n,n);
    for i=1:n,
        for j=1:n,
            %Cov(i,j) = K_square_exp(alpha,beta, x(i), x(j)); 
            %Cov(i,j) = K_linear(alpha,x(i), x(j)); 
            %Cov(i,j) = K_Wiener_process(alpha,x(i), x(j)); 
            Cov(i,j) = K_absolute(alpha,x(i), x(j)); 
        end
    end
    
    %Select functions from Gaussian Process
    [U,S,V] = svd(Cov);
    Z = U*sqrt(S)*X;
   
    %Plot
    figure(3); hold on; 
    plot(x,Z,'r.-');
    axis([0,1,-2,2]);
    title('GP with absolute Exp Kernel');

end


function [cov_ij] = K_square_exp(alpha,beta,x,y),
    arg = x-y;
    component = arg'*arg /(beta*beta*2);
    cov_ij = alpha*alpha*exp(-component); 
end

function [cov_ij] = K_linear(alpha,x,y),
    cov_ij = alpha*x*y;
end

function [cov_ij] = K_Wiener_process(alpha,x,y),
    cov_ij = alpha*min(x,y);
end

function [cov_ij] = K_absolute(alpha,x,y),
    arg = x-y;
    cov_ij = alpha*alpha * exp(-abs(arg));
end
