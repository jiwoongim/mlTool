##Regressions

I implemented two regression method, [Polynomial Regression](https://raw.github.com/jiwoongim/mlTool/master/regressions/polyRegression.py)
and [Bayesian Curive Fitting with Evidence Approximation](https://raw.github.com/jiwoongim/mlTool/master/regressions/bayesianCurveFit_EvidenceApproximation.py).
For the simple testing purpose, I generated toydata from degree 3 polynomial with gaussian noise added. I Sampled 60
points, and selected random 40 points as train dat and rest of 20 points as test data. 

##Polynomial Regression

After running 10trials with different amount of train and test set.
```
#Train/#Test |  LSDE PolyReg on Train | LSDE Bay on Train | LSDE on PolyReg Test | LSDE on Bay Test

50 / 50    |    7.0232 +- 0.1781    |  7.0237 +- 0.1780 |    7.1729 +- 0.2173  |   7.1701 +- 0.22
```


Here are two plots that was generated using polynomial regression.
![Training](https://raw.github.com/jiwoongim/mlTool/master/Regressions/images/polyRegTrain.png)
![Testing](https://raw.github.com/jiwoongim/mlTool/master/Regressions/images/polyRegTest.png)


##Bayesian Curve fitting with evidence approximation
![Training](https://raw.github.com/jiwoongim/mlTool/master/Regressions/images/bayesianTrain.png)
![Testing](https://raw.github.com/jiwoongim/mlTool/master/Regressions/images/bayesianTest.png)

The further derivations and details of two methods can be view at 
[regressions.pdf](https://github.com/jiwoongim/mlTool/blob/master/Regressions/regressions.pdf)

