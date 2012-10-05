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


'''Computing the weight w, given data point x,y and polynomial degree K'''
def polyRegression(K, x, y):

    A = np.zeros([len(x), K]);
   
    #building matrix
    for i in range(K):
        for j in range(len(x)):
            A[j,i] = np.power(x[j], i);
  
    invA = np.linalg.pinv(A);
    w = invA.dot(np.array(y));
    print w
    return w;

'''Evaluating polynomial given polynomial degree K, data value xt, and 
weight w'''
def NPolynomReg(K, xt, t, beta):

    yt = np.zeros(len(xt));
    for i in range(len(xt)):
        for j in range(K):

            #compute #y(x_i, w)
            yt[i] = np.power(xt[i], j) * w[j] + yt[i];
    
    for i in range(len(xt)):
        #sample from gaussian distribution N(t_i| y_i, beta);
        eps = utils.gaussian(yt[i], t[i], beta); 
        p = np.random.random();
        if p >= 0.5:
            yt[i] = yt[i] + utils.gaussian(yt[i], t[i], beta); 
        else :
            yt[i] = yt[i] - utils.gaussian(yt[i], t[i], beta); 

    return yt;

'''Evaluating polynomial given polynomial degree K, data value xt, and 
weight w'''
def evalPolynomReg(K, xt, w):
    yt = np.zeros(len(xt));
    for i in range(len(xt)):
        for j in range(K):
            yt[i] = np.power(xt[i], j) * w[j] + yt[i];

    return yt;


def updateWeight(K, xTrain, yTrain):

    X = basisX(xTrain);
    X = np.matrix(X);
    #print np.linalg.pinv(X).shape,  X.dot(np.array(yTrain)).shape
    w = np.linalg.pinv(X).dot((X.dot(yTrain)).T);
    return w;

def basisX(xTrain):
    X = np.zeros([K, len(xTrain)]);
    for i in range(K):
        for j in range(len(xTrain)):
            X[i,j] = pow(xTrain[j], i);
    return X

def updateVariance(oTrain, yTrain):

    var = np.sqrt(np.sum(np.power((oTrain-yTrain),2)))/ float(len(oTrain));
    return var;
    #beta = np.linalg.norm(oTrain- yTrain) / float(len(oTrain));
    #return beta;

if __name__ == '__main__':

    K = 4;
    kList = [3];
    errorTrain = [0]*len(kList);
    errorTest = [0]*len(kList);
    
    #data
    [X, Y, T] = dt.sample_poly3(60);
    [xTrain, yTrain, tTrain, xTest, yTest, tTest] = util.sepTrainTest4Reg(40,20,\
            X,T,Y);

    #maximum likelihood
    w = polyRegression(K, xTrain, yTrain);  #weight computed 
    #oTrain = evalPolynomReg(K, xTrain, w);  
    #oTrain2 = oTrain;
   
    oTrain = evalPolynomReg(K, xTrain, w);  
    oTrain2 = oTrain;
    beta = updateVariance(oTrain, yTrain);
    print beta

    #evaluating train
    oTrain = NPolynomReg(K, xTrain, yTrain, beta);
    error = np.linalg.norm(np.array(yTrain)-oTrain);
    
    #evaluating test
    oTest = NPolynomReg(K, xTest, yTest, beta);
    errorTest[0] = np.linalg.norm(np.array(oTest) -yTest);    

    #graph
    pl.figure(1);
    t = pl.plot(xTrain, yTrain, 'go');
    o = pl.plot(xTrain, oTrain, 'ro-');
    pl.plot(xTrain, oTrain2, 'bo-');
    #pl.legend((t, o), ("target", "predicted"));
    #pl.ylim([-8,5]);
    pl.title('Polynomial Regression train set with K=' + str(K));

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



