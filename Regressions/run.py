import polyRegression as pR
import probPolyReg as ppR
import bayesianCurveFit as bcf;
import pylab as pl;
import regTestData as dt;
import numpy as np;
from mpl_toolkits.mplot3d import Axes3D
import sys;

sys.path.insert(0, '../dataSet');

import util;


def ntrial(numTrial, numTrain, numTest):
    for i in range(numTrial):
        #data
        [X, Y, T] = dt.sample_poly3(numTrain+numTest);
        [xTrain, yTrain, tTrain, xTest, yTest, tTest] = util.sepTrainTest4Reg(numTrain,numTest,\
                X,T,Y);

        errTrainPoly, errTestPoly = pR.run (xTrain, yTrain, tTrain, xTest, yTest, tTest, 4);
        #errTrainPoly, errTestPoly = ppR.run(xTrain, yTrain, tTrain, xTest, yTest, tTest, 4);
        errTrainBay, errTestBay = bcf.run(xTrain, yTrain, tTrain, xTest, yTest, tTest, 4);
        errPoly[0].append(round(errTrainPoly,6));
        errPoly[1].append(round(errTestPoly,6)); 
        errBay[0].append(round(errTrainBay,6));
        errBay[1].append(round(errTestBay,6));

    return errPoly, errBay;

if __name__ == '__main__':

    errBay = [[],[]];
    errPoly = [[],[]];
    errPoly, errBay = ntrial(8, 50, 50);

    print "Euclidean error distance for train: "
    print errPoly[0]
    print errBay[0]
    print np.mean(errPoly[0]), np.var(errPoly[0])
    print np.mean(errBay[0]), np.var(errBay[0])

    print "Euclidean error distance for test: "
    print errPoly[1]
    print errBay[1]
    print np.mean(errPoly[1]), np.var(errPoly[1])
    print np.mean(errBay[1]), np.var(errBay[1])

    #pl.show();

