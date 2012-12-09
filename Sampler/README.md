## Sampling Methods

I implemented two simpling method, [MCMC Metropolis-Hasting] (https://raw.github.com/jiwoongim/mlTool/master/Sampler/MCMC.py)
and [Gibbs Sampling] (https://raw.github.com/jiwoongim/mlTool/master/Sampler/GibbsSampling.py).
For the simple testing purpose, I target distribution was 2d multivariate Gaussian distribution.
I sampled 500, 1000, 10000 points using MCMC and Gibb Sampling.

###MCMC metropolis-Hasting
Here are three 2d histogram that was sampled from MCMC.
![500MCMC](https://raw.github.com/jiwoongim/mlTool/master/Sampler/image/500samples_mcmc.png)
![1000MCMC](https://raw.github.com/jiwoongim/mlTool/master/Sampler/image/1000samples_mcmc.png)
![10000MCMC](https://raw.github.com/jiwoongim/mlTool/master/Sampler/image/10000samples_mcmc.png)

###Gibbs Sampling
Here are three 2d histogram that was sampled from Gibbs Sampling.
![500gibbs](https://raw.github.com/jiwoongim/mlTool/master/Sampler/image/500samples_gibbs.png)
![1000gibbs](https://raw.github.com/jiwoongim/mlTool/master/Sampler/image/1000samples_gibbs.png)
![10000gibss](https://raw.github.com/jiwoongim/mlTool/master/Sampler/image/10000samples_gibbs.png)


