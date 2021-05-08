import numpy as np
from Gaussians import *
from Objective import *

def OneStep(Archetypes,Transformation,BarycenterPenalty,ArchetypePenalty,SampleSize, NumberOfIterations):
    NumberOfAtoms  = len(Archetypes[0])
    NumberOfArchetypes =len(Archetypes)
    PlanSize = NumberOfAtoms ** 2
    TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize


    MeanMatrix = np.array(np.random.rand(TotalDimension))
    CovMatrix = np.identity(TotalDimension)

    for i in range(NumberOfIterations):
        Samples = SampleGeneration(MeanMatrix, CovMatrix, SampleSize)
        LogLikelihoodValues = LogLikelihood(Samples,(Archetypes,Transformation,BarycenterPenalty,ArchetypePenalty))
        Weights             = np.exp(-LogLikelihoodValues)
        MeanMatrix = np.array(GaussianReconstruction(Samples, Weights)[0])
        CovMatrix  = np.array(GaussianReconstruction(Samples, Weights)[1])
        Factor     = np.min(np.diagonal(CovMatrix))
        CovMatrix  = CovMatrix/Factor
        print("The minimum loglikelihood value is ",np.min(LogLikelihoodValues),"\n")
        print("The mean barycenter is  ", np.exp(MeanMatrix[:9])/np.sum(np.exp(MeanMatrix[:9])),"\n","\n")

