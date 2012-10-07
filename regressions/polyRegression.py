import numpy as np;
import pylab as pl;
import matplotlib;
import matplotlib.pyplot as plt
import sys;
from mpl_toolkits.mplot3d import Axes3D



sys.path.insert(0, '../dataSet');
import regTestData as dt;
import util;


def sigmoid(x):
    return 1/np.exp(-x);

'''Computing the weight w, given data point x,y and polynomial degree K'''
def polyRegression(K, x, y):

    A = np.zeros([len(x), K]);
   
    #building matrix
    for i in range(K):
        for j in range(len(x)):
            A[j,i] = np.power(x[j], i);
  
    invA = np.linalg.pinv(A);
    w = invA.dot(np.array(y));
    return w;

'''Evaluating polynomial given polynomial degree K, data value xt, and 
weight w'''
def evalPolynomReg(K, xt, w):
    yt = np.zeros(len(xt));
    for i in range(len(xt)):
        for j in range(K):
            yt[i] = np.power(xt[i], j) * w[j] + yt[i];

    return yt;


def run(xTrain, yTrain, tTrain, xTest, yTest, tTest, K):

    kList = [3]#kList = range(1,8);
    errorTrain = [0]*len(kList);
    errorTest = [0]*len(kList);
    K = 4;
    #for K in kList:
    w = polyRegression(K, xTrain, yTrain);  #weight computed

    #evaluating train
    oTrain = evalPolynomReg(K, xTrain, w);  
    #errorTrain[0] = np.linalg.norm(np.array(yTrain)-oTrain);
    print "Euclidean error distance for train: "
    print np.linalg.norm(np.array(yTrain)-oTrain);
 
    #evaluating test
    oTest = evalPolynomReg(K, xTest, w);
    #errorTest[0] = np.linalg.norm(np.array(oTest) -yTest);
    print "Euclidean error distance for test: "
    print np.linalg.norm(np.array(oTest) -yTest);    


    pl.figure(1);
    t = pl.plot(xTrain, yTrain, 'go', label="train data");
    o = pl.plot(xTrain, oTrain, 'ro-', label="predicted data");
    pl.legend((t, o), ("target", "predicted"));
    pl.title('Polynomial Regression train set with degree ' + str(K-1));

    pl.figure(2);
    t = pl.plot(xTest, yTest, 'go', label="test data");
    o = pl.plot(xTest, oTest, 'ro', label="predicted");
    pl.legend();
    pl.title('Polynomial Regression test set with degree ' + str(K-1));

if __name__ == '__main__':

    #data
    [X, Y, T] = dt.sample_poly3(60);
    [xTrain, yTrain, tTrain, xTest, yTest, tTest] = util.sepTrainTest4Reg(20,10,\
            X,T,Y);

    polyRegFlag = True;
    if polyRegFlag :   

        kList = [3]#kList = range(1,8);
        errorTrain = [0]*len(kList);
        errorTest = [0]*len(kList);
        K = 4;
        #for K in kList:
        w = polyRegression(K, xTrain, yTrain);  #weight computed
    
        #evaluating train
        oTrain = evalPolynomReg(K, xTrain, w);  
        errorTrain[0] = np.linalg.norm(np.array(yTrain)-oTrain);

        #evaluating test
        oTest = evalPolynomReg(K, xTest, w);
        errorTest[0] = np.linalg.norm(np.array(oTest) -yTest);
        
        pl.figure(1);
        t = pl.plot(xTrain, yTrain, 'go', label="train data");
        o = pl.plot(xTrain, oTrain, 'r-', label="predicted data");
        pl.legend((t, o), ("target", "predicted"));
        pl.title('Polynomial Regression train set with degree ' + str(K-1));

        pl.figure(2);
        t = pl.plot(xTest, yTest, 'go', label="test data");
        o = pl.plot(xTest, oTest, 'ro', label="predicted");
        pl.legend();
        pl.title('Polynomial Regression test set with degree ' + str(K-1));

        #pl.figure(K+1);
        #etr = pl.plot(kList, errorTrain, 'gx');
        #etst = pl.plot(kList, errorTest, 'rx');
        #pl.legend((etr, etst), ("train error", "test error"));
        #pl.title('Least Square Error');
    pl.show();
