## Restricted Boltzmann Machine

1. Implemented RBM, Gaussian RBM, RBM with Rectified Linear Units.
2. Implemented Discriminative RBM.

### 1. RBM, Gaussian RBM, RBM with Rectified Linear Units
I experimented those three models by training subet of MNIST data, and generating image from
the three models.
I used 400 hidden units for all three models. From the generated images
from each model, I could see their characters/behaviour.

### RBM
![rbm] (https://raw.github.com/jiwoongim/mlTool/master/RBMs/image/gensubdata_rbm1.jpg)
### Gaussian RBM
![grbm] (https://raw.github.com/jiwoongim/mlTool/master/RBMs/image/genSubdata_gauss.jpg)
### RBM with Rectified Linear Units
![grbm] (https://raw.github.com/jiwoongim/mlTool/master/RBMs/image/genSubdata%20rlu.jpg)

More details can be found in [rbm.pdf] (https://github.com/jiwoongim/mlTool/blob/master/RBMs/rbm.pdf)

### 2. Discriminative RBM

The discriminative RBM has interconnection between hidden layer with input layer, and also hidden layer with class label 
layers. Similar to the diagram shown:\\

![drbm](https://raw.github.com/jiwoongim/mlTool/master/RBMs/image/classrbm.jpg)

