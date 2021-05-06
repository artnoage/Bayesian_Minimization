import numpy as np

MeanMatrix         = [0,0]
CovMatrix          = [[0.5,0.5],[0.5,0.5]]
samplesize         = 10
data = np.random.multivariate_normal(MeanMatrix,CovMatrix,samplesize)
x=data[:,0]
y=data[:,1]
w=-30*((y-x)**2)
w=np.exp(w)
print(x-y)
mu = np.ma.average(data, axis=0, weights=w)
Sigma = np.cov((data).T, bias=True, aweights=w)

print(Sigma)