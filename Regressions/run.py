import polyRegression as pR
import probPolyReg as ppR
import bayesianCurveFit as bcf;
import pylab as pl;
import regTestData as dt;
from mpl_toolkits.mplot3d import Axes3D
import sys;

sys.path.insert(0, '../dataSet');

import util;

if __name__ == '__main__':

    #data
    [X, Y, T] = dt.sample_poly3(60);
    [xTrain, yTrain, tTrain, xTest, yTest, tTest] = util.sepTrainTest4Reg(40,20,\
            X,T,Y);

    pR.run (xTrain, yTrain, tTrain, xTest, yTest, tTest, 4);
    ppR.run(xTrain, yTrain, tTrain, xTest, yTest, tTest, 4);
    bcf.run(xTrain, yTrain, tTrain, xTest, yTest, tTest, 4);
    pl.show();

