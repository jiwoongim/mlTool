import numpy;
import scipy;
import pylab as pl;
from matplotlib import pyplot as plt

import sys;
sys.path.insert(0, '../dataSet');
import noise_image as ni

def computeEnergy(img, denoiseImg, row, column):
   
    #init coefficient
    beta =  1;
    alpha = 2.1;
    gamma = 0.3;

    #bias
    bias = sum(sum(denoiseImg));

    #cliques of the form x_i y_i
    term1 = sum(sum(img*denoiseImg));

    #cliques comprise pairs of varible x_i x_j
    term2 = 2*sum(sum(denoiseImg[1:,:]*denoiseImg[:row-1,:])) \
            + 2*sum(sum(denoiseImg[:,1:]*denoiseImg[:,:column-1]));

    energy = alpha*bias - gamma*term1 - beta*term2;
    return energy;

def computePartition(img, row, column, denoiseImg):

    energy = 0;
    for i in range(row):
        for j in range(column):
            #for k in range(row):
            #    for h in range(column):
                    tmp1 = img;
                    tmp2 = denoiseImg;
   

                    for state in [-1,1]:
                        tmp1[i,j] = state;                    
                        #tmp2[k,h] = -1;
                        energy += computeEnergy(tmp1, tmp2, row, column);
                        print i,j, energy
                        #tmp2[k,h] = 1;
                        #energy += computeEnergy(tmp1, tmp2, row, column);


    print energy


def iteratedConditionalModes(img, row, column, denoiseImg):
   
    shuffleInd_x = numpy.arange(row);
    shuffleInd_y = numpy.arange(column);

    numpy.random.shuffle(shuffleInd_x);
    numpy.random.shuffle(shuffleInd_y);
    for k in range(1):
        for i in range(row):
            for j in range(column):
        #for i in range(row):
        #    for j in range(column):


                tmp = denoiseImg;
                tmp[shuffleInd_x[i], shuffleInd_y[j]] = -1;
                eng1 = computeEnergy(tmp, denoiseImg, row, column);
                
                tmp[shuffleInd_x[i], shuffleInd_y[j]] = 1;
                eng2 = computeEnergy(tmp, denoiseImg, row, column);
                
                if (eng1 < eng2):
                    denoiseImg[shuffleInd_x[i],shuffleInd_y[j]] = -1;
                else:
                    denoiseImg[shuffleInd_x[i],shuffleInd_y[j]] = 1;
            print i
    
    
    pl.figure(2);
    pl.imshow(denoiseImg);#[row/4:row/3, column/4:column/3]);
    pl.savefig("restoredImg.jpg");
    #scipy.misc.imsave("restoredImg.jpg", image_array);   
    return denoiseImg;

if __name__ == "__main__":
   
    img, denoiseImg, row, column = ni.noiseImage1("../dataSet/");
    print row, column 
    pl.figure(1);
    pl.imshow(img);#[row/4:row/2, column/4:column/2]);

    iteratedConditionalModes(img, row, column, denoiseImg);
    #print computeEnergy(img, denoiseImg, row, column);
    #eng = computePartition(img, row, column, denoiseImg);

    plt.show(); 


