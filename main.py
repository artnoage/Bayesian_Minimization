import numpy as np
from VisualisationBarycenter import *
from Algorithm import *
from Objective import *

NumberOfIterations=1
SampleSize=50000

#Provide the Images
Archetypes   = np.array([[0.3, 0.3,0.3,0.01,0.01,0.02,0.02,0.02,0.02], [0.02,0.02,0.02,0.02,0.01,0.01,0.3,0.3,0.3],[0.3, 0.3,0.1,0.01,0.01,0.02,0.2,0.02,0.02], [0.02,0.02,0.02,0.02,0.01,0.3,0.01,0.3,0.3]])


#Itterating Algorithm
#Transformation is Normalized,Exponential,Normalized Exponential and No Transformation. Penalty types are Square and Entropy.

for i in range(1):
    Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#    Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="Entropy", ArchetypePenalty="Entropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#    Barycenter=OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="RevEntropy", ArchetypePenalty="RevEntropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="RevEntropy", ArchetypePenalty="Entropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#OneStep(Archetypes, TransformationFunction="Normalized Exponential", BarycenterPenalty="Entropy", ArchetypePenalty="RevEntropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#   OneStep(Archetypes, TransformationFunction="Normalized", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#    Barycenter=OneStep(Archetypes, TransformationFunction="Normalized", BarycenterPenalty="RevEntropy", ArchetypePenalty="RevEntropy", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)
#OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations)

#Visualisation
ani=visualise(Archetypes,Barycenter)
plt.show()