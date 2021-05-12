import numpy as np
from VisualisationBarycenter import *
from Algorithm import *
from Objective import *

NumberOfIterations=10000
SampleSize=1000

#Provide the Images
Archetypes   = np.array([[0.3, 0.3,0.3,0.01,0.01,0.02,0.02,0.02,0.02], [0.02,0.02,0.02,0.02,0.01,0.01,0.3,0.3,0.3]])
NumberOfAtoms = len(Archetypes[0])
NumberOfArchetypes = len(Archetypes)
PlanSize = NumberOfAtoms ** 2
TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize
StepSize=10**(-4)


#Initializing Mean and Covariance

MeanMatrixInitialization = np.array(np.random.rand(TotalDimension))
CovMatrixInitialization = np.identity(TotalDimension)

#with open('Square.npy', 'rb') as g:
#    MeanMatrixInitialization=np.load(g)
#    CovMatrixInitialization=np.load(g)


#Itterating Algorithm
#Transformation is Normalized,Exponential,Normalized Exponential and No Transformation. Penalty types are Square and Entropy.


MeanMatrix, CovMatrix =OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="Square", ArchetypePenalty="Square",PriorType="Gaussian",
                              SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization,StepSize=StepSize)

#MeanMatrix, CovMatrix =OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="RevEntropy", ArchetypePenalty="RevEntropy",PriorType="Gaussian",
# SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,CovMatrixInitialization=CovMatrixInitialization,StepSize=StepSize)
print(MeanMatrix)

with open('Square.npy', 'wb') as f:
    np.save(f,MeanMatrix)
    np.save(f,CovMatrix)

#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
#Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="Entropy", ArchetypePenalty="Entropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="RevEntropy", ArchetypePenalty="RevEntropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="RevEntropy", ArchetypePenalty="Entropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="Entropy", ArchetypePenalty="RevEntropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#Barycenter=OneStep(Archetypes, TransformationFunction="Normalized", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#Barycenter=OneStep(Archetypes, TransformationFunction="Normalized", BarycenterPenalty="RevEntropy", ArchetypePenalty="RevEntropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)