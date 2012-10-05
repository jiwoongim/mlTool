import numpy as np;
import utils; 
import pylab as pl;
import matplotlib;
import matplotlib.pyplot as plt
import sys;
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '../dataSet');
import regTestData as dt;
import util;
def basisX(xTrain):
    X = np.zeros([K, len(xTrain)]);
    for i in range(K):
        for j in range(len(xTrain)):
            X[i,j] = pow(xTrain[j], i);
    return X


def bayesianCurveFitting(xTrain, yTrain):
   
    #init
    alpha = 0.01;
    beta = 1;
    X = basisX(xTrain).T; 
    N = X.shape[1]; #number of data points
    invS = alpha*np.identity(K) + beta*X.T.dot(X);
    S = np.linalg.inv(invS); #Covariance Matrix
    mean = beta*S.dot(X.T).dot(yTrain); #Mean
    std = 1/beta  + X.dot(S).dot(X.T); #standard deviation


    flag = True; count = 0;
    while flag:

        #Computing Eigen values of Covariance matrix
        eigVals = np.linalg.eigvals(S);

        #Gamma 
        for eigVal in eigVals:
            gamma = eigVal / (eigVal + alpha);

        #Error
        err = computeErr(yTrain, X, mean, N); 
        
        #Update
        preAlpha = alpha;
        alpha = gamma / mean.T.dot(mean);
        beta = (N-gamma)/err;

        if (count >=5 and abs(alpha -preAlpha) < 0.001):
            flag = False;
        print "Updated alpha, belta: ", str(alpha) +' '+ str(beta)
        count += 1;

    print "Number of iteration: " + str(count);
    oTrain = eval1(yTrain, X, mean, N);
    return oTrain, mean;

def eval1(y, X, mean, N):
    output = []
    for ind in range(y.shape[0]): 
        output.append(mean.dot(X[ind,:]));
    return output; 

def computeErr(yTrain, X, mean, N):
    err = 0;
    for ind in range(yTrain.shape[0]): 
        err += yTrain[ind] - mean.dot(X[ind,:]);
    
    return err;

if __name__ == '__main__':

    K = 4;
    kList = [3];
    errorTrain = [0]*len(kList);
    errorTest = [0]*len(kList);
    
    #data
    [X, Y, T] = dt.sample_poly3(60);
    [xTrain, yTrain, tTrain, xTest, yTest, tTest] = util.sepTrainTest4Reg(40,20,\
            X,T,Y);

    oTrain, mean = bayesianCurveFitting(xTrain, np.array(yTrain));
    
    #graph
    pl.figure(1);
    t = pl.plot(xTrain, yTrain, 'go');
    o = pl.plot(xTrain, oTrain, 'ro-');
    #pl.plot(xTrain, oTrain2, 'bo-');
    #pl.legend((t, o), ("target", "predicted"));
    pl.ylim([-8,5]);
    pl.title('Bayesian Regression train set with K=' + str(K));

    Basis = basisX(xTest); 
    N = Basis.shape[1];
    oTest = eval1(np.array(yTest), Basis.T, mean, N); 
    pl.figure(2);
    t = pl.plot(xTest, yTest, 'go');
    o = pl.plot(xTest, oTest, 'rx');
    #pl.legend((t,o), ("target", "predicted"));
    pl.ylim([-8,5]);
    pl.title('Polynomial Regression test set with K=' + str(K));

    #pl.figure(K+1);
    #etr = pl.plot(kList, errorTrain, 'gx');
    #etst = pl.plot(kList, errorTest, 'rx');
    #pl.legend((etr, etst), ("train error", "test error"));
    #pl.title('Least Square Error');
    pl.show();



