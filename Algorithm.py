import numpy as np
from Gaussians import *
from Objective import *

def OneStep(MeanMatrix,CovMatrix,SampleSize,LogLikelihood,NumberOfIterations):
    for i in range(1000):
        MeanMatrixOut=[]
        for j in range(int(NumberOfIterations/1000)):
            Samples = SampleGeneration(MeanMatrix, CovMatrix, SampleSize)
            LogLikelihoodValues = LogLikelihood(Samples)
            Weights             = np.exp(-LogLikelihoodValues)
            MeanMatrixOut.append(np.array(GaussianReconstruction(Samples, Weights)[0]))
        MeanMatrix = np.average(MeanMatrixOut,axis=0)
        CovMatrix  = np.array(GaussianReconstruction(Samples, Weights)[1])
        factor     = np.min(np.diagonal(CovMatrix))
        CovMatrix  = CovMatrix/factor
        print(np.min(LogLikelihoodValues))
        print(np.exp(MeanMatrix[:9])/np.sum(np.exp(MeanMatrix[:9])))

