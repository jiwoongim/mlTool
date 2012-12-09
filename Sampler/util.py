import numpy as np;
import scipy as sp;
import pylab as pl;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import random;
import math;


def histogram2d(fig,samples):
    #ax = fig.add_subplot(111,projection='3d');
    hist, xedges,yedges = np.histogram2d(samples[0,:], samples[1,:], bins=(40,40));
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]];
    plt.imshow(hist, extent=extent, interpolation='nearest')
    plt.colorbar();
 
def scatter_plot(fig, samples):
    plt.scatter(samples[0,:], samples[1,:], marker='d', c='r');
    plt.title('Gibbs Sampling');
    plt.grid();

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


