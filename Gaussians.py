import numpy as np
from TransformationsAndPenalties import *

def SampleGeneration(PriorType,MeanMatrix,CovMatrix,samplesize):
    if PriorType=="Gaussian":
        data = np.random.multivariate_normal(MeanMatrix,CovMatrix,samplesize).astype("float64")
    return data

#Recover Mean and covariance based on Data

def GaussianReconstruction(PriorType,Samples,Weights):
    if PriorType=="Gaussian":
        MeanMatrix = np.ma.average(Samples, axis=0, weights=Weights)
        CovMatrix = np.cov(Samples.T,aweights=Weights)
    return MeanMatrix, CovMatrix


