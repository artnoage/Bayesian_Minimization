from Loglikelihood import *
from Gaussians import *

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
            MeanMatrixInitialization=np.load(g)
    elif MeanMatrix=="New":
        MeanMatrixInitializationPart1=np.ones(NumberOfAtoms)/NumberOfAtoms
        MeanMatrixInitializationPart2=np.ones((NumberOfAtoms**2)*NumberOfArchetypes)/(NumberOfAtoms**2)
        MeanMatrixInitialization=np.concatenate((MeanMatrixInitializationPart1,MeanMatrixInitializationPart2),axis=0)
    if CovMatrix=="Load":
        with open('Square.npy', 'rb') as g:
            CovMatrixInitialization=np.load(g)
    elif CovMatrix=="New":
        CovMatrixInitialization = Factor*np.identity(TotalDimension)
    return MeanMatrixInitialization, CovMatrixInitialization


def OneStep(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, PriorType, SampleSize, NumberOfIterations, MeanMatrixInitialization, CovMatrixInitialization, LoglikelihoodFactor):
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
    Factor = np.max(np.diag(CovMatrix))

    for i in range(NumberOfIterations):

        # Here we generate the Samples and calculate the  weights

        Samples = SampleGeneration(PriorType, MeanMatrix, CovMatrix, SampleSize)
        LogLikelihoodValues = LogLikelihood(Samples, (Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty))

        print("The minimum loglikelihood value is ",
              np.mean(LogLikelihoodValues[LogLikelihoodValues.argsort()[-50:][::-1]]), "\n")
        LoglikelihoodValuesNormalized=LoglikelihoodFactor*LogLikelihoodValues

        Weights = np.exp(-LoglikelihoodValuesNormalized)

        #  Here we find the best fit for Mean and Covariance.
        MeanMatrix = np.array(GaussianReconstruction(PriorType, Samples, Weights)[0])
        CovMatrix  = np.array(GaussianReconstruction(PriorType, Samples, Weights)[1])

        #If we like, we normalize a bit.
        CovNormal="Yes"
        if CovNormal=="Yes":
            CovMatrix=Factor*(CovMatrix/np.max(np.diag(CovMatrix)))


        if i%100==99:
            Barycenter = np.array(Transformation([MeanMatrix[0:NumberOfAtoms]], Inputtype="Barycenter",
                                                 Transformationfunction=TransformationFunction))
            print("The mean barycenter is  ", Barycenter ,  "\n")

        if i%500==499:
            with open('Square.npy', 'wb') as f:
                np.save(f, MeanMatrix)
                np.save(f, CovMatrix)

    return MeanMatrix, CovMatrix
