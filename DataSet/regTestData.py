import numpy as np;
import scipy as sp;
import pylab as pl;
import matplotlib.pyplot as plt;
import numpy.matlib;
from mpl_toolkits.mplot3d import Axes3D


#global parameter
numPoints = 30;

#data
def sample_poly3(numPoints):
    
    #INIT poly3 coef
    a = 1;
    b = -7;

    numPoints = numPoints/10;
    xL = range(1, 10*numPoints, 1) + [numPoints*10]
    yL = [0]*len(xL);
    tL = [0]*len(xL);
    xA = np.array(xL)/float(numPoints);

    for i in range(len(xL)):
        xd = xA[i];
        x = xd-3;
        epsilon = numpy.matlib.randn(1);
        tL[i] = 0.1*(a* pow(x,3) + b*x*x + x);
        yL[i] = tL[i] + epsilon[0,0];

       
    #pl.plot(list(xA), yL, 'bo');
    #pl.plot(list(xA), tL, 'ro-');

    return [list(xA), yL, tL];

if __name__ == '__main__':
    
    #initializing
    #fig = plt.figure(1)
    #fig.clf()
    #ax = Axes3D(fig);
    
    #X = sample_data1();
    sample_poly3(numPoints);
    pl.show();

