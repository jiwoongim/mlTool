import toydata 
import numpy.matlib as ml;
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
    projX = base[:,0:dimR].dot(projX) #+ base[:,dimR:].dot(bias);  #projected data.

    return projX

# evaluate
def evaluate_em(dimR, base, eVal, X, ax2):

    threshold = 1e-4;
    N = X.shape[0];
    K = X.shape[1];
    xMean = np.mean(X, axis=1).reshape(3,1); #mean
    W = ml.rand(N, dimR);    # covariance
    Z = W.T.dot(X-xMean);  #recon
    sigma_2 = sum(sum(np.power((W.dot(Z) - X), 2)).T) / (N*dimR); #variance
    sigma_2 = sigma_2[0,0];
    M = W.T.dot(W) + sigma_2*np.ones([dimR, dimR]); 

    iteration = 1;
    oldProbX = 1;
    flag = True;
    while flag:
        #E-step 
        EZ = np.linalg.inv(M).dot(W.T.dot(X - xMean));   #expectation of latent variable
        EZZ = sigma_2*np.linalg.inv(M) + EZ.dot(EZ.T);

        #M-step
        W = N* (X-xMean).dot(EZ.T).dot(N*np.linalg.inv(EZZ));
        sigma_2 = (np.power(np.linalg.norm(X-xMean),2) + np.trace(EZZ.T.dot(W.T.dot(W))) +\
                2*(sum(sum(EZ.T.dot(W.T.dot(X-xMean))).T))) / (N * dimR);
        sigma_2= sigma_2[0,0];

        iteration += 1;
        invM = np.linalg.inv(M);
        probX = N*K + K*(N*np.log(sigma_2) + np.trace(invM) \
                  - np.log(np.linalg.det(invM))) + np.trace( EZ.T.dot(EZ));
        err = abs(1- probX/oldProbX);
        oldProbX = probX;
        if (iteration > 8 and err < threshold):
            flag = False;

           

    # transformation matrix to principal space 
    Cov = W.dot(W.T) + sigma_2*np.ones([W.shape[0], W.shape[0]]);
    [base1, eVal, order] = compute_eig(Cov, dimR);

    #[base1, eVal, order] = compute_eig(np.cov(W.T.dot(X-xMean)), dimR);
    eVec = base1[:, :dimR];
    W = base1.dot(W);
    Z = W.T.dot(X-xMean);

    #projecting train data
    projX = base1[:,0:dimR].T.dot(X);
    xMean = np.mean(X, axis=1).reshape(3,1);
    tmp = xMean *np.ones(X.shape[1]*3).reshape(3,X.shape[1]);
    bias = base1[:,dimR:].T.dot(tmp);
    projX = base1[:,0:dimR].dot(projX) + base1[:,dimR:].dot(bias);  #projected data.
    
    #visualize projected Data
    #ax2.plot(projX[0,:], projX[1,:], projX[2,:], 'o');# label='data projected into principal space');
    ax2.plot(projX[0,:Z.shape[1]/2], projX[1,:Z.shape[1]/2], projX[2,:Z.shape[1]/2], 'ro');# label='data projected into principal space');
    ax2.plot(projX[0,Z.shape[1]/2:], projX[1,Z.shape[1]/2:], projX[2,Z.shape[1]/2:], 'bo');# label='data projected into principal space');
 
    print "The number of EM steps iterated: " + str(iteration);
    return [base1, Z]

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
    W, Z = evaluate_em(dimR, base, eVal, X, ax2);  #projecting trains data and test data
    
    #visualize eigenvectors
    eVec1 = np.array((np.array([0,0,0]), base[:,0]*(eVal[0]*scale1)));
    eVec2 = np.array((np.array([0,0,0]), base[:,1]*(-eVal[1]*scale2)));
    ax.plot(eVec1[:,0], eVec1[:,1], eVec1[:,2], 'r-', label='principal axes');
    ax.plot(eVec2[:,0], eVec2[:,1], eVec2[:,2], 'r-');


    w1 = np.array((np.array([0,0,0]), W[:,0]*(-eVal[0]*scale1)));
    w2 = np.array((np.array([0,0,0]), W[:,1]*(eVal[1]*0.01)));#*scale2)));
    ax.plot(w1[:,0], w1[:,1], w1[:,2], 'b-', label='');
    ax.plot(w2[:,0], w2[:,1], w2[:,2], 'g-');

    ax2.plot(w1[:,0], w1[:,1], w1[:,2], 'b-', label='');
    ax2.plot(w2[:,0], w2[:,1], w2[:,2], 'g-');  
    
    ax.legend();
    ax2.legend();    
   
    
    if (vis1 ==1):
        pl.show();
    

    



