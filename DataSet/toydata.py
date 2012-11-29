import numpy as np
import scipy as sp
import pylab as pl

#global parameter
numPoints = 120;    #number of sample points 

#data
def sample_data1(ax):
    
    X = np.random.rand(3,numPoints);
    xtrain =X[:, :numPoints * 2/3];
    xt = X[:, numPoints *2/3 :];
    xMean = np.mean(xtrain, axis=1);

    xtrain.sort(axis=1);
    xt.sort(axis=1);
    x1 = xtrain[:,:(xtrain.shape[1]/2) ];
    x2 = xtrain[:,(xtrain.shape[1]/2) :];

    x = np.concatenate((x1, x2), axis=1);

    #visualizing the toy data
    ax.plot(x1[0,:], x1[1,:], x1[2,:], 'o', label='class1 train data');
    ax.plot(x2[0,:], x2[1,:], x2[2,:], 'o', label='class2 train data');
    ax.plot([xMean[0]], [xMean[1]],  [xMean[2]], 'o', label='mean');
    return x, xt, X 

def sample_data_exp(ax, k):

    # sampling points from standard exponential distribution
    X1 = np.random.standard_exponential((3, numPoints));
    X2 = k- np.random.standard_exponential((3, numPoints));
    X = np.concatenate((X1,X2), axis=1);

    x1 = X1[:,:X1.shape[1] * 2/3];
    xt1 = X1[:, X1.shape[1] * 2/3 :];
    x2 = X2[:,:X2.shape[1] * 2/3];
    xt2 = X2[:, X2.shape[1] * 2/3 :];

    x = np.concatenate((x1,x2), axis =1);
    Xt = np.concatenate((xt1, xt2), axis=1);
    xMean = np.mean(X, axis=1);

    #visualizing the toy data
    #if (vis1 ==1):
    ax.plot(x1[0,:], x1[1,:], x1[2,:], 'o', label='class1 train data');
    ax.plot(x2[0,:], x2[1,:], x2[2,:], 'o', label='class2 train data');
    ax.plot([xMean[0]], [xMean[1]],  [xMean[2]], 'o', label='mean');

    return x, Xt, X 


