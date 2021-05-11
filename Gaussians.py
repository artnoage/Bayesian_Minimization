import numpy as np

def SampleGeneration(PriorType,MeanMatrix,CovMatrix,samplesize):
    data = np.random.multivariate_normal(MeanMatrix,CovMatrix,samplesize).astype("float64")
    return data

#Recover Mean and covariance based on Data

def GaussianReconstruction(PriorType,Samples,Weights):
    MeanMatrix = np.ma.average(Samples, axis=0, weights=Weights)
    CovMatrix = np.cov(Samples.T,aweights=Weights)
    return MeanMatrix, CovMatrix


