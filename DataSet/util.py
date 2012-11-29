import numpy as np;
import scipy as sp;
import pylab as pl;
import sys;
import matplotlib.pyplot as plt;
import numpy.matlib;
import random;
from mpl_toolkits.mplot3d import Axes3D

def sepTrainTest4Reg(nTrain, nTest, X,T,Y):

    nTot = nTrain + nTest;   
    if (nTot != len(X)):
        print " number of train + number of test data doesn't match" \
                + "number of total data"
        sys.exit(0);

    xTest =[]
    tTest = []; 
    yTest = []; 
    events = range(0,nTot);
    for i in range(nTest):
        r = int(random.random() * nTot);
        xTest.append(X.pop(r));
        yTest.append(Y.pop(r));
        tTest.append(T.pop(r));
        nTot -=1;
    xTrain = X;
    yTrain = Y;
    tTrain = T;

    return [xTrain, yTrain, tTrain, xTest, yTest, tTest];


        


         
        



