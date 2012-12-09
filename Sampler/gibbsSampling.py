import numpy as np;
import util;
import scipy as sp;
import pylab as pl;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import random;
import math;

def gibbs_sampler(numSamples):

    #Initize 
    dim = 2  #Experimenting with 2-dimension
    samples = np.zeros([dim,numSamples]);
    #samples[:,0] = np.array([-4, -3]); 
    samples[:,0] = np.array([random.random(), random.random()]); 
    #samples[:,0] = np.array([math.floor(random.random()*10), math.floor(random.random())*10]); 

    for j_sample in range(1, numSamples): #For each samples
        samples[0, j_sample] = condProb(samples[1, j_sample-1]); #x1 ~ N(x1|x2,1)
        samples[1, j_sample] = condProb(samples[0, j_sample]);  #x2 ~ N(x2|x1,1)

    return samples

def condProb(X):
    corr = 0.5;
    std = 1-corr**2; 
    var = np.sqrt(std);
    return var*np.random.randn() + X*corr;
    return X*corr + np.random.randn();

def condProb1(X):
    return -1 + (X-1)/4 + 15/4*(np.random.randn());
def condProb2(X):
    return 1 + (X+1)/4 + 15/4*(np.random.randn());

   
if __name__ == '__main__':
    
    print "P(x1|x2) ~ N(0, 1)";
    print "P(x2|x1) ~ N(0, 1)";

    plotflag = [ 500, 1000, 10000];

    #Gibb Sampling
    for i_iter in plotflag:
        samples = gibbs_sampler(1000); 

        fig = pl.figure(i_iter);
        util.histogram2d(fig,samples);
        #scatter_plot(fig, samples)
        #histogram(fig, samples);

    plt.show()

