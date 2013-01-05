import numpy as np;
import scipy as sp;
import pylab as pl;
import matplotlib.pyplot as plt;
import random;
import math;

from mpl_toolkits.mplot3d import Axes3D;

'''square exponential kernal function.
x,y  - input vectors
alpha, beta - hyperparameters'''
def K_square_exp(x,y, alpha, beta):
 
    argument = x-y;
    #comp1 = (1/(beta*beta)) * argument.T.dot(argument); 
    comp1 =  argument *argument /(beta*beta*2); 
    k = alpha*alpha * np.exp(-comp1);
    return k;
 
'''linear kernal function.
x,y  - input vectors
alpha, beta - hyperparameters'''
def K_linear(x,y, alpha, beta): 

    return x*y; 

'''Gaussian Process'''
def main(alpha, beta):

    #Select uniform samples from the interval
    k = 1000;
    x = np.array(range(0,k,5))/float(k);
    n = x.shape[0];
    X = np.random.randn(n);

    #Compute covariance matrix
    Cov = np.zeros([n,n]);

    for i in range(n):
        for j in range(n):
            Cov[i,j] = K_square_exp(X[i],X[j],alpha,beta);
            #Cov[i,j] = K_linear(X[i],X[j],alpha,beta);

    print Cov

    #Select functions from Gaussian Process
    U,S,V = np.linalg.svd(Cov);
    Z = U.dot(np.multiply(np.sqrt(S), X));

    #Plot
    pl.figure(6);
    t = pl.plot(x, Z, 'g.-');
    pl.ylim([-5,5]);
    #pl.legend((t,o), ("target", "predicted"));
    pl.title('Gaussian Process');


if __name__ == '__main__':

    print "Gaussian Process: Demo for Various Types of Kernels"

    #hyperparameter 
    alpha = 1;
    beta = 1/5.0;

    main(alpha, beta);
    pl.show();

    #t = np.array([700, 800, 1025]);
    ##Computer covariance matrix
    #Cov = np.zeros([3,3]);

    #for i in range(3):
    #    for j in range(3):
    #        Cov[i,j] = K_square_exp(t[i],t[j],7,50);

    #print Cov



