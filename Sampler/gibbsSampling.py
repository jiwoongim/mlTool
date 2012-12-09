import numpy as np;
import scipy as sp;
import pylab as pl;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import random;
import math;

def gibbs_sampler(numSamples, plotflag):

    #Gibb Sampling
    for i_iter in range(plotflag[-1]+1):
        #Initize 
        dim = 2  #Experimenting with 2-dimension
        samples = np.zeros([dim,numSamples]);
        #samples[:,0] = np.array([-4, -3]); 
        samples[:,0] = np.array([random.random(), random.random()]); 
        #samples[:,0] = np.array([math.floor(random.random()*10), math.floor(random.random())*10]); 

        for j_sample in range(1, numSamples): #For each samples
            samples[0, j_sample] = condProb(samples[1, j_sample-1]); #x1 ~ N(x1|x2,1)
            samples[1, j_sample] = condProb(samples[0, j_sample]);  #x2 ~ N(x2|x1,1)

        if (i_iter in plotflag):
            print i_iter
            fig = pl.figure(i_iter);
            histogram2d(fig,samples);
            #scatter_plot(fig, samples)
            #histogram(fig, samples);

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


def scatter_plot(fig, samples):
    plt.scatter(samples[0,:], samples[1,:], marker='d', c='r');
    plt.title('Gibbs Sampling');
    plt.grid();

def histogram2d(fig,samples):
    #ax = fig.add_subplot(111,projection='3d');
    hist, xedges,yedges = np.histogram2d(samples[0,:], samples[1,:], bins=(40,40));
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]];
    plt.imshow(hist, extent=extent, interpolation='nearest')
    plt.colorbar();
    
def histogram(fig,samples):
    ax = fig.add_subplot(111,projection='3d');
    hist, xedges,yedges = np.histogram2d(samples[0,:], samples[1,:], bins=1000);
    
    elements = (len(xedges)-1) * (len(yedges)-1);
    xpos, ypos = np.meshgrid(xedges[:-1]+0.3, yedges[:-1]+0.3); 

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements);
    dx = 0.3 * np.ones_like(zpos);
    dy = dx.copy()
    dz = hist.flatten()

    surf = ax.bar3d(xpos, ypos, zpos,dx,dy,dz);
    #fig.colorbar(surf, shrink=0.5, aspect=10);

if __name__ == '__main__':
    
    print "P(x1|x2) ~ N(-1+(x2-1)/4, 1)";
    print "P(x2|x1) ~ N( 1+(x1+1)/4, 1)";

    plotflag = [ 500, 1000, 10000];
    gibbs_sampler(1000, plotflag);
    plt.show()

