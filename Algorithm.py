import numpy as np
from Gaussians import *
from Objective import *

def OneStep(MeanMatrix,CovMatrix,SampleSize,Likelihood,NumberOfIterations):

    for i in range(NumberOfIterations):

        Samples = SampleGeneration(MeanMatrix, CovMatrix, SampleSize)
        LikelihoodValues = Likelihood(Samples)
        Weights = np.exp(-LikelihoodValues)
        MeanMatrix = np.array(GaussianReconstruction(Samples, Weights)[0])
        CovMatrix = np.array(GaussianReconstruction(Samples, Weights)[1])
        print(np.max(LikelihoodValues), np.min(LikelihoodValues))
        print(MeanMatrix[:4]/np.sum(MeanMatrix[:4]))
    #print(np.exp(MeanMatrix[:4])/(np.sum(np.exp(MeanMatrix[:4]))))
