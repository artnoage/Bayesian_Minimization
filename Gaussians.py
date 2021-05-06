import numpy as np

def SampleGeneration(MeanMatrix,CovMatrix,samplesize):
    data = np.random.multivariate_normal(MeanMatrix,CovMatrix,samplesize).astype("float64")
    return data

#Recover Mean and covariance based on Data

def GaussianReconstruction(s,w):
    mu = np.ma.average(s, axis=0, weights=w)
    Sigma = np.cov(s.T,aweights=w)
    return mu, Sigma


