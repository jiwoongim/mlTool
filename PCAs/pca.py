import toydata 
import numpy as np
import scipy as sp
import pylab as pl
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#global parameter
dimR =2;   #number of princial components
vis1 = 1;   #vis is set to 1 to view the graphs
scale1 = 1;
scale2 = 1;

#Computing eigenvalues and eigenvectors
def compute_eig(x, dimR):
    xMean = np.mean(x, axis=1);
    tmp = x- np.tile(np.reshape(xMean,[xMean.shape[0],1]), x.shape[1]);
    covM = tmp.dot(np.transpose(tmp));
    [eVal, eVec] = np.linalg.eig(covM);

    order = np.argsort(eVal, axis=0);
    order = order[::-1];
   
    eVal = eVal[:,order];
    base = eVec[:,order];
    #base = base[:,0:dimR];
    
    return [base,eVal, order];

#project data
def project(x, dimR, base, X):
    projX = base[:,0:dimR].T.dot(x);
    xMean = np.mean(X, axis=1).reshape(3,1);
    tmp = xMean *np.ones(x.shape[1]*3).reshape(3,x.shape[1]);
    bias = base[:,dimR:].T.dot(tmp);
    projX = base[:,0:dimR].dot(projX) + base[:,dimR:].dot(bias);  #projected data.

    return projX

# evaluate
def evaluate(x, xt, dimR, base, X, ax2):

    #projecting train data
    projX = project(x, dimR, base, X);

    #projecting test data
    projXt = project(xt, dimR, base, X);
   
    #1-distanced 
    Y = scipy.spatial.distance.cdist(projX.T, projXt.T, 'euclidean'); 
    xt1 = xt[:, np.argmin(Y, axis=0)<= x.shape[1]/2];
    xt2 = xt[:, np.argmin(Y, axis=0)> x.shape[1]/2];
    projXt1 = projXt[:, np.argmin(Y, axis=0)<= x.shape[1]/2];
    projXt2 = projXt[:, np.argmin(Y, axis=0)> x.shape[1]/2];

    #visualize projected Data
    ax2.plot(projX[0,:x.shape[1]/2], projX[1,:x.shape[1]/2], projX[2,:x.shape[1]/2], 'o', label='class1 train data projected ');
    ax2.plot(projX[0,x.shape[1]/2:], projX[1,x.shape[1]/2:], projX[2,x.shape[1]/2:],'o', label='class2 train data projected ');
    ax2.plot(projXt1[0,:], projXt1[1,:], projXt1[2,:], 'o', label='class1 test data projected ');
    ax2.plot(projXt2[0,:], projXt2[1,:], projXt2[2,:], 'o', label='class2 test data projected ');

    return [xt1, xt2];

if __name__ == "__main__":

    #initializing
    fig = plt.figure(1)
    fig.clf()
    ax = Axes3D(fig);

    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = Axes3D(fig2);

    #trainData, testData, X = toydata.sample_data_exp(ax, 5); scale1 = 0.001; scale2 = 0.01;
    trainData, testData, X = toydata.sample_data1(ax); scale1= 0.1;   #building sample data
    [base, eVal, order] = compute_eig(trainData, dimR); #computing eigenvalues, eigenvectors
    [xt1, xt2] = evaluate(trainData, testData, dimR, base, X, ax2);  #projecting trains data and test data
    
    #visualize trainset
    ax.plot(xt1[0,:], xt1[1,:], xt1[2,:], 'o', label='class1 test data');
    ax.plot(xt2[0,:], xt2[2,:], xt2[2,:], 'o', label='class2 test data');

    #visualize eigenvectors
    print eVal;
    eVec1 = np.array((np.array([0,0,0]), base[:,0]*(-eVal[0]*scale1)));
    eVec2 = np.array((np.array([0,0,0]), base[:,1]*(eVal[1]*scale2)));
    ax.plot(eVec1[:,0], eVec1[:,1], eVec1[:,2], 'r-', label='eigenvector');
    ax.plot(eVec2[:,0], eVec2[:,1], eVec2[:,2], 'r-');

    ax.legend();
    ax2.legend();    
   
    
    if (vis1 ==1):
        pl.show();
    

    



