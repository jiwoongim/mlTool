## Neural Network

I implemented neural network, and experimented classification error with various features. The features includes: 
momentum, adpative learning rate for each connection units, drop out. For the practice/quick experiement 
purpose, I used subset of MNIST data where I only consider digits 2,3,4. Also, I set the network to have one 
hidden layer with 400 hidden units. The hypothesis was that everytime I add the features to the network, the 
performance on the test set will get better.\\

###Procedure
```
1. Ran a regular neural network with just backpropagation.
2. Ran the neural network with momentum and adapative learning rates/connections
3. Ran the neural network with momentum, adapative learning rates, and drop outs.
```

###Results
```
                       Train  Test
###NN                | 0.102 | 0.073
###NN_mom_adpt       | 0.068 | 0.063
###NN_mom_adpt_drop  | 0.071 | 0.0465
```

## Neural Network 
