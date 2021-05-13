import numpy as np

def SampleGeneration(PriorType,MeanMatrix,CovMatrix,samplesize):
    if PriorType=="Gaussian":
        data = np.random.multivariate_normal(MeanMatrix,CovMatrix,samplesize).astype("float64")
    elif PriorType=="LogNormal":
        data = np.random.multivariate_normal(MeanMatrix, CovMatrix, samplesize).astype("float64")
        data=np.exp(data)
    return data

#Recover Mean and covariance based on Data

def GaussianReconstruction(PriorType,Samples,Weights):
    if PriorType=="Gaussian":
        MeanMatrix = np.ma.average(Samples, axis=0, weights=Weights)
        CovMatrix = np.cov(Samples.T,aweights=Weights)
    elif PriorType=="LogNormal":
        MeanMatrix = np.log(np.ma.average(Samples, axis=0, weights=Weights))-0.5
        CovMatrix = np.log(np.cov(Samples.T, aweights=Weights)/np.tensordot(MeanMatrix,MeanMatrix,axes=0)+1)
    return MeanMatrix, CovMatrix


