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
def project(x, dimR, base):
    projX = base[:,0:dimR].T.dot(x);
    xMean = np.mean(X, axis=1).reshape(3,1);
    tmp = xMean *np.ones(x.shape[1]*3).reshape(3,x.shape[1]);
    bias = base[:,dimR:].T.dot(tmp);
    projX = base[:,0:dimR].dot(projX) + base[:,dimR:].dot(bias);  #projected data.

    return projX

# evaluate
def evaluate(dimR, base, eVal, X, ax):

    print base.shape
    # variance 
    sigma_2 = 1/(X.shape[1]-dimR) * sum(eVal[dimR:]);

    # transformation matrix to principal space 
    U = base[:, :dimR];
    L = eVal[:dimR]*np.identity(dimR);
    W = U.dot(np.sqrt(L-sigma_2*np.identity(dimR))); 

    xMean = np.mean(X, axis=1).reshape(3,1);
    Cov = np.cov(W.T.dot(X-xMean));
    [eVal1, eVec1] = np.linalg.eig(Cov);
    order = np.argsort(eVal1, axis=0);
    order = order[::-1];
    
    eVal1 = eVal1[:,order];
    eVec1 = eVec1[:,order];

    W = W.dot(eVec1);
    Z = W.T.dot(X-xMean);

    #projecting train data
    projX = project(X, dimR, base);
    
    #visualize projected Data
    print projX.shape, Z.shape[1]/2
    ax.plot(projX[0,:Z.shape[1]/2], projX[1,:Z.shape[1]/2], projX[2,:Z.shape[1]/2], 'o', label='data projected into principal space');
    ax.plot(projX[0,Z.shape[1]/2:], projX[1,Z.shape[1]/2:], projX[2,Z.shape[1]/2:], 'o', label='data projected into principal space');

    return [W, Z]

if __name__ == "__main__":

    #initializing
    fig = plt.figure(1)
    fig.clf()
    ax = Axes3D(fig);

    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = Axes3D(fig2);

    trainData, testData, X = toydata.sample_data_exp(ax, 5); scale1 = 0.001; scale2 = 0.01;
    #trainData, testData, X = toydata.sample_data1(ax); scale1= 0.1;   #building sample data
    [base, eVal, order] = compute_eig(trainData, dimR); #computing eigenvalues, eigenvectors
    W, Z = evaluate(dimR, base, eVal, X, ax2);  #projecting trains data and test data
    
    #visualize eigenvectors
    eVec1 = np.array((np.array([0,0,0]), base[:,0]*(-eVal[0]*scale1)));
    eVec2 = np.array((np.array([0,0,0]), base[:,1]*(eVal[1]*scale2)));
    ax.plot(eVec1[:,0], eVec1[:,1], eVec1[:,2], 'r-', label='principal axes');
    ax.plot(eVec2[:,0], eVec2[:,1], eVec2[:,2], 'r-');

    w1 = np.array((np.array([0,0,0]), W[:,0]*(-eVal[0]*scale1*scale1)));
    w2 = np.array((np.array([0,0,0]), W[:,1]*(eVal[1]*scale2*scale2)));
    ax.plot(w1[:,0], w1[:,1], w1[:,2], 'b-', label='');
    ax.plot(w2[:,0], w2[:,1], w2[:,2], 'g-');


    ax.legend();
    ax2.legend();    
   
    
    if (vis1 ==1):
        pl.show();
    

    



