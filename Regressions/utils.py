import numpy as np;
import math;
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


    
def gaussian(x,mean, var):
    #print  math.exp( -(pow(x-mean/var,2))), var
    term1 = (x-mean)/var;
    term = -(pow(term1,2)/2);
    term2 = var* math.sqrt(2*math.pi);
    return 10*math.exp(term) / term2;


if __name__ == '__main__':
    print gaussian(1,4, 4.0)*10;

