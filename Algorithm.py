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
        LogLikelihoodValues = (10 ** (-5)) * LogLikelihood(Samples, (
        Archetypes, Transformation, BarycenterPenalty, ArchetypePenalty))
        Weights = np.exp(-LogLikelihoodValues)

        for j in range(20):
            Sample = SampleGeneration(MeanMatrix, CovMatrix, SampleSize)
            LogLikelihoodValue = (10**(-5))*LogLikelihood(Sample,(Archetypes,Transformation,BarycenterPenalty,ArchetypePenalty))
            Weight             = np.exp(-LogLikelihoodValue)

            order=Weight.argsort()
            Sample=Sample[order[::-1]]
            Weight=Weight[order[::-1]]
            Sample=Sample[:np.int(SampleSize/20)]
            Weight=Weight[:np.int(SampleSize/20)]
            Samples=np.concatenate((Sample,Samples))
            Weights=np.concatenate((Weight,Weights))

        MeanMatrix = np.array(GaussianReconstruction(Samples, Weights)[0])
        CovMatrix  = np.array(GaussianReconstruction(Samples, Weights)[1])
        Factor     = np.min(np.diagonal(CovMatrix))
        CovMatrix  = CovMatrix/Factor
    print("The minimum loglikelihood value is ",np.min((10**5)*LogLikelihoodValue),"\n")
    print("The mean barycenter is  ", np.exp(MeanMatrix[:9])/np.sum(np.exp(MeanMatrix[:9])),"\n","\n")

