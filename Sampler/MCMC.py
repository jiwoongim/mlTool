import numpy as np;
import util;
import scipy as sp;
import pylab as pl;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import random;
import math;

def MCMC(numIter, numSamples, target):

    #Initalize
    count = 1; 
    cov = np.min(target[1])**2; #Random walk behaviour (step size) 
    samples = np.zeros([2,numSamples]);
    samples[:,0] = target[0] + np.random.randn(2)*cov; #start point

    while count != numSamples:
        x = np.random.randn(2)*cov + samples[:,count-1];
        prev_prob = normal(target, samples[:,count-1]);
        prob_x = normal(target, x);
        alpha = min(prob_x/prev_prob,1); #Accped with pro0ability alpha
        uni = np.random.random();
        if (uni <= alpha):  #accept
            samples[:,count] = x;
            count += 1;

    return samples; 

def normal(target,x):
    X = (x-target[0]);
    invCov = np.linalg.inv(target[1]); 
    detCov = np.linalg.det(target[1]);
    p = 1/(2*np.pi*np.sqrt(detCov)) * np.exp(-0.5*(np.dot(np.dot(X.T, invCov),X)));
    return p;

if __name__ == '__main__':
    print "Target Distribution is"
    print "mean : (0,0)"
    print "variance (2, 0.5)"
    
    mean = np.array([0,0]);
    var = np.array([[2,0.5],[0.5,2]]);
    target = [mean, var];

    plotflag = [500,1000,10000];
    #MCMC 
    for i_iter in plotflag:
        samples = MCMC(1000, 1000, target);

        #Plot
        fig = pl.figure(i_iter);
        util.histogram2d(fig,samples);


    plt.show();
