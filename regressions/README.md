##Regressions

I implemented two regression method, [Polynomial Regression](https://raw.github.com/jiwoongim/mlTool/master/regressions/polyRegression.py)
and [Bayesian Curive Fitting with Evidence Approximation](https://raw.github.com/jiwoongim/mlTool/master/regressions/bayesianCurveFit_EvidenceApproximation.py).
For the simple testing purpose, I generated toydata from degree 3 polynomial with gaussian noise added. I Sampled 60
points, and selected random 40 points as train dat and rest of 20 points as test data. 

##Polynomial Regression

Here are two plots that was generated using polynomial regression.
![Training](https://raw.github.com/jiwoongim/mlTool/master/regressions/images/polyRegTrain.png)
![Testing](https://raw.github.com/jiwoongim/mlTool/master/regressions/images/polyRegTest.png)


##Bayesian Curve fitting with evidence approximation
![Training](https://raw.github.com/jiwoongim/mlTool/master/regressions/images/bayesianTrain.png)
![Testing](https://raw.github.com/jiwoongim/mlTool/master/regressions/images/bayesianTest.png)

The further derivations and details of two methods can be view at 
[regressions.pdf](https://github.com/jiwoongim/mlTool/blobmaster/regressions/regressions.pdf)

