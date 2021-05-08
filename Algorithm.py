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
        #factor= np.min(np.diagonal(CovMatrix))
        #CovMatrix=CovMatrix/factor
