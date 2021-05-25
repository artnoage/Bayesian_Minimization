import cupy as cp
import numpy as np
from TransformationsAndPenalties import *

def SampleGeneration(PriorType,MeanMatrix,CovMatrix,samplesize):
    if PriorType=="Gaussian":
        data = cp.random.multivariate_normal(MeanMatrix,CovMatrix,samplesize,check_valid="warn",method="eigh",dtype=np.float64)
    return data

#Recover Mean and covariance based on Data

def GaussianReconstruction(PriorType,Samples,Weights):
    if PriorType=="Gaussian":
        Samples = cp.asnumpy(Samples)
        Weights = cp.asnumpy(Weights)
        MeanMatrix = np.average(Samples, axis=0, weights=Weights)
        CovMatrix = np.cov(Samples.T,aweights=Weights)
        MeanMatrix=cp.array(MeanMatrix)
        CovMatrix=cp.array(CovMatrix)
    return MeanMatrix, CovMatrix


