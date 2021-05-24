from Loglikelihood import *
import cupy as cp
from Gaussians import *
import time as time

def initialisation(Archetypes, MeanMatrix, CovMatrix, Factor):

    '''Here we decide if we want to start with a new Mean and Covariance Matrix or if we want to load the old ones
    For new mean we create one that uses the uniform measures for both barycenter and plan. For the covariance matrix
    we multiply with a constant depending on if we want to localize.'''

    NumberOfAtoms = len(Archetypes[0])
    NumberOfArchetypes = len(Archetypes)
    PlanSize = NumberOfAtoms ** 2
    TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize

    if MeanMatrix=="Load":
        with open('Square.npy', 'rb') as g:
            MeanMatrixInitialization=cp.load(g)
    elif MeanMatrix=="New":
        MeanMatrixInitializationPart1= cp.ones(NumberOfAtoms) / NumberOfAtoms
        MeanMatrixInitializationPart2= cp.ones((NumberOfAtoms ** 2) * NumberOfArchetypes) / (NumberOfAtoms ** 2)
        MeanMatrixInitialization=cp.concatenate((MeanMatrixInitializationPart1, MeanMatrixInitializationPart2), axis=0)
    if CovMatrix=="Load":
        with open('Square.npy', 'rb') as g:
            CovMatrixInitialization=cp.load(g)
    elif CovMatrix=="New":
        CovMatrixInitialization = Factor * cp.identity(TotalDimension)
    return MeanMatrixInitialization, CovMatrixInitialization


def OneStep(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, ReluPenalty, PriorType, SampleSize, NumberOfIterations, MeanMatrixInitialization, CovMatrixInitialization, NormalizeCovariance, LoglikelihoodFactor):
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
    Factor = cp.max(cp.diag(CovMatrix))
    args=(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, ReluPenalty)
    LoglikelihoodA=  Loglikelihood1(args).Loglikelihoodsimple
    LoglikelihoodB = Loglikelihood2(args).Loglikelihoodsimple
    StartTime=0
    for i in range(NumberOfIterations):

        # Here we generate the Samples and calculate the  weights

        Samples = SampleGeneration(PriorType, MeanMatrix, CovMatrix, SampleSize)

        executionTime = (time.time() - StartTime)
        print('Execution time in seconds: ' + str(executionTime))
        StartTime = time.time()
        LogLikelihoodValues = LoglikelihoodA(Samples)


        #print("The minimum loglikelihood value is ",
        #      np.mean(LogLikelihoodValues[LogLikelihoodValues.argsort()[-100:][::-1]]), "\n")

        LoglikelihoodValuesNormalized=LoglikelihoodFactor*LogLikelihoodValues

        Weights = cp.exp(-LoglikelihoodValuesNormalized)

        #  Here we find the best fit for Mean and Covariance.
        MeanMatrix, CovMatrix = GaussianReconstruction(PriorType, Samples, Weights)


        #If we like, we normalize a bit.
        if NormalizeCovariance=="Yes":
            CovMatrix=Factor*(CovMatrix / cp.max(cp.diag(CovMatrix)))


        if i%100==99:
            Barycenter = cp.array(Transformation([MeanMatrix[0:NumberOfAtoms]], Inputtype="Barycenter",
                                                 Transformationfunction=TransformationFunction))
            print("The mean barycenter is  ", Barycenter ,  "\n")

        if i%500==499:
            with open('Square.npy', 'wb') as f:
                cp.save(f, MeanMatrix)
                cp.save(f, CovMatrix)

    return MeanMatrix, CovMatrix
