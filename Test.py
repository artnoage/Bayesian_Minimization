import numpy as np
import itertools
from Gaussians import *

def ObjectiveRelu(Data,PartitionLength,Archetypes):
    NumberOfAtoms = PartitionLength ** 2
    PlanSize = NumberOfAtoms ** 2
    NumberOfArchetypes=len(Archetypes)
    T = np.linspace(0, 1, PartitionLength)
    couples = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]

    CostMatrix = np.linalg.norm(a - b, axis=1) ** 2
    CostMatrix = CostMatrix + np.zeros((len(Data), NumberOfArchetypes,1))
    Archetypes = Archetypes + np.zeros((len(Data), NumberOfArchetypes,1))


    Barycenter         = np.array([Data[:,:NumberOfAtoms],Data[:,:NumberOfAtoms]]).transpose(1,0,2)
    Plans              = Data[:,NumberOfAtoms:].reshape(-1,NumberOfArchetypes,PlanSize)
    TransportationCost = np.sum(np.sum(CostMatrix[:,:,:]*Plans[:,:,:],axis=2),axis=1)



    BarycenterMargin  = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms).transpose(0,1,3,2),axis=3)
    ArchetypeMargin   = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms),axis=3)
    BarycenterPenalty = np.sum(np.linalg.norm(BarycenterMargin -Barycenter,axis=2),axis=1)
    ArchetypePenalty  = np.sum(np.linalg.norm(ArchetypeMargin -Archetypes,axis=2),axis=1)
    ReluData = np.sum(np.minimum(Data - 0.0001, 0), axis=1)
    TotalCost = TransportationCost + 5 * ArchetypePenalty + 5 * BarycenterPenalty + np.exp(-ReluData)
    return TotalCost

def ObjectiveRelu2(Data,PartitionLength,Archetypes):
    NumberOfAtoms = PartitionLength ** 2
    PlanSize = NumberOfAtoms ** 2
    NumberOfArchetypes=len(Archetypes)
    T = np.linspace(0, 1, PartitionLength)
    couples = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]

    CostMatrix = np.linalg.norm(a - b, axis=1) ** 2
    CostMatrix = CostMatrix + np.zeros((len(Data), NumberOfArchetypes,1))
    Archetypes = Archetypes + np.zeros((len(Data), NumberOfArchetypes,1))

    Data               = np.exp(Data)
    Barycenter         = np.array([Data[:,:NumberOfAtoms],Data[:,:NumberOfAtoms]]).transpose(1,0,2)
    Barycenter         = Barycenter*(1/np.sum(Barycenter,axis=2).reshape(-1,2,1))
    Plans              = Data[:,NumberOfAtoms:].reshape(-1,NumberOfArchetypes,PlanSize)
    Plans              = Plans*(1/np.sum(Plans.transpose(0,2,1),axis=1).reshape(-1,NumberOfArchetypes,1))
    TransportationCost = np.sum(np.sum(CostMatrix[:,:,:]*Plans[:,:,:],axis=2),axis=1)



    BarycenterMargin  = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms).transpose(0,1,3,2),axis=3)
    ArchetypeMargin   = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms),axis=3)
    BarycenterPenalty = np.sum(np.linalg.norm(BarycenterMargin -Barycenter,axis=2),axis=1)
    ArchetypePenalty  = np.sum(np.linalg.norm(ArchetypeMargin -Archetypes,axis=2),axis=1)
    ReluData = np.sum(np.minimum(Data - 0.0001, 0), axis=1)
    TotalCost = TransportationCost + 5 * ArchetypePenalty + 5 * BarycenterPenalty
    print(TotalCost)
    return TotalCost




PartitionLength    =3
Archetypes1        = np.array([[1, 0, 0,0,0,0,0,0,0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])
Archetypes         = Archetypes1
NumberOfArchetypes = len(Archetypes)






