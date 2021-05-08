import numpy as np
from Visualisation import *
from Algorithm import *
from Objective import *



#Provide the Images
Archetypes   = np.array([[0.92, 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01], [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.92]])


#Itterating Algorithm
OneStep(Archetypes,Transformation="Normalized",BarycenterPenalty="Entropy",ArchetypePenalty="Entropy",SampleSize=5*10**5,NumberOfIterations=10)


#Visualisation
#ani=visualise(Objective,MeanMatrices,CovMatrices)
#plt.show()