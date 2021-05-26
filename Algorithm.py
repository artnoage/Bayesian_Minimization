import numpy as np

from Loglikelihood import *
import cupy as cp
from Gaussians import *
import time as time

def initialisation(Archetypes, MeanMatrix, CovMatrix, Precision):

    '''Here we decide if we want to start with a new Mean and Covariance Matrix or if we want to load the old ones
    For new mean we create one that uses the uniform measures for both barycenter and plan. For the covariance matrix
    we multiply with a constant depending on if we want to localize.'''

    NumberOfAtoms = len(Archetypes[0])
    NumberOfArchetypes = len(Archetypes)
    PlanSize = NumberOfAtoms ** 2
    TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize
    with open('Square.npy', 'rb') as g:
        A=cp.load(g)
        B=cp.load(g)
        C=cp.load(g)
        if MeanMatrix=="Load":
            MeanMatrixInitialization=A
        elif MeanMatrix=="New":
            MeanMatrixInitializationPart1= cp.ones(NumberOfAtoms) / NumberOfAtoms
            MeanMatrixInitializationPart2= cp.ones((NumberOfAtoms ** 2) * NumberOfArchetypes) / (NumberOfAtoms ** 2)
            MeanMatrixInitialization=cp.concatenate((MeanMatrixInitializationPart1, MeanMatrixInitializationPart2), axis=0)
        if  CovMatrix=="Load":
            CovMatrixInitialization=B
        if  Precision=="Load":
            Precision=C
        if  CovMatrix=="New":
            CovMatrixInitialization = Precision*cp.identity(TotalDimension)
    return MeanMatrixInitialization, CovMatrixInitialization, Precision


def OneStep(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, ReluPenalty, PriorType, SampleSize, SamplingNumber, NumberOfIterations, MeanMatrixInitialization, CovMatrixInitialization, NormalizeCovariance, ForcedPrecision,
            LoglikelihoodFactor):
    """This is the main Algorithm. In every iteration it samples from the previous Gaussian, passes the samples through
    the likelihood and then calculates the mean and covariance for the Gaussian that fits the weighted samples the most.
    It saves the Mean and Covariance in the end, in case you want to rerun the code with something modified. There is also
    a part where one can play with the localisation of the covariance."""


    NumberOfAtoms  = len(Archetypes[0])
    NumberOfArchetypes = len(Archetypes)
    PlanSize = NumberOfAtoms ** 2
    TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize
    MeanMatrix = MeanMatrixInitialization
    CovMatrix = CovMatrixInitialization
    args=(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, ReluPenalty)
    Loglikelihood=  Objectives(args).Cost

    StartTime=0
    for i in range(NumberOfIterations):
        Samples=SampleGeneration(PriorType, MeanMatrix, CovMatrix, SampleSize)
        LoglikelihoodValues=Loglikelihood(Samples)
        # Here we generate the Samples and calculate the  weights

        for j in range(SamplingNumber):
            TempSamples = SampleGeneration(PriorType, MeanMatrix, CovMatrix, SampleSize)
            TempLogLikelihoodValues = Loglikelihood(TempSamples)
            Samples=cp.concatenate((Samples,TempSamples),axis=0)
            LoglikelihoodValues=cp.concatenate((LoglikelihoodValues,TempLogLikelihoodValues),axis=0)
            Best=LoglikelihoodValues.argsort()[:SampleSize]
            Samples=cp.take(Samples,Best,axis=0)
            LoglikelihoodValues=cp.take(LoglikelihoodValues,Best,axis=0)

        print(cp.min(LoglikelihoodValues))

        LoglikelihoodValuesNormalized=LoglikelihoodValues*LoglikelihoodFactor
        Weights = cp.exp(-LoglikelihoodValuesNormalized)


        #  Here we find the best fit for Mean and Covariance.
        MeanMatrix, CovMatrix = GaussianReconstruction(PriorType, Samples, Weights)
        Precision=cp.max(cp.diag(CovMatrix))


        #If we like, we normalize a bit.
        if NormalizeCovariance=="Yes":
            CovMatrix=ForcedPrecision*(CovMatrix / Precision)


        if  i%20==19:
            print("The minimum loglikelihood value is ",
                  cp.mean(LoglikelihoodValues[LoglikelihoodValues.argsort()[-50:][::-1]]), "\n")
            Barycenter = cp.array(Transformation(cp.array(MeanMatrix[0:NumberOfAtoms]), Inputtype="Barycenter",
                                                 Transformationfunction=TransformationFunction))
            print("The mean barycenter is  ", Barycenter ,  "\n")
            print("Saving...", "The Precision is", Precision)
            with open('Square.npy', 'wb') as f:
                cp.save(f, MeanMatrix)
                cp.save(f, CovMatrix)
                cp.save(f,Precision)

    return MeanMatrix, CovMatrix, Precision
