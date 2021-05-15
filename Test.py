import numpy as np

Archetypes   = np.array([[0.3, 0.3,0.3,0.01,0.01,0.02,0.02,0.02,0.02], [0.02,0.02,0.02,0.02,0.01,0.01,0.3,0.3,0.3]])
NumberOfAtoms = len(Archetypes[0])
NumberOfArchetypes = len(Archetypes)
PlanSize = NumberOfAtoms ** 2
TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize
A1=np.ones(NumberOfAtoms)/NumberOfAtoms
A2=np.ones((NumberOfAtoms**2)*NumberOfArchetypes)/(NumberOfAtoms**2)
A3=np.concatenate((A1,A2),axis=0)
print(A3)