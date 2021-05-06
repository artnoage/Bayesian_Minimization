import numpy as np
import itertools

def ObjectiveRelu(Data,PartitionLength,Archetypes):
    NumberOfAtoms = PartitionLength ** 2
    PlanSize = NumberOfAtoms ** 2

    T = np.linspace(0, 1, PartitionLength)
    couples = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]
    CostMatrix = np.linalg.norm(a - b, axis=1) ** 2

    CostMatrix = CostMatrix + np.zeros((len(Data), 1))

    SampleOfBarycenter = np.array([Datum[0:4] for Datum in Data])
    SampleOfBarycenter = np.array([Barycenter / np.sum(Barycenter) for Barycenter in SampleOfBarycenter])
    SampleOfFirstPlan = np.array([Datum[4:20] for Datum in Data])
    SampleOfFirstPlan = np.array([FirstPlan / np.sum(FirstPlan) for FirstPlan in SampleOfFirstPlan])
    SampleOfSecondPlan = np.array([Datum[20:36] for Datum in Data])
    SampleOfSecondPlan = np.array([SecondPlan / np.sum(SecondPlan) for SecondPlan in SampleOfSecondPlan])

    SampleOfFirstTransportationCost = np.sum(CostMatrix * SampleOfFirstPlan, axis=1)
    SampleOfSecondTransportationCost = np.sum(CostMatrix * SampleOfSecondPlan, axis=1)
    TransportationCost = SampleOfFirstTransportationCost + SampleOfSecondTransportationCost



    SampleOfFirstBarycenterMargin = np.array([np.sum(FirstPlan.reshape(4, 4).T, axis=1) for FirstPlan in SampleOfFirstPlan])
    SampleOfFirstArchetypeMargin = np.array([np.sum(FirstPlan.reshape(4, 4), axis=1) for FirstPlan in SampleOfFirstPlan])
    SampleOfSecondBarycenterMargin = np.array([np.sum(SecondPlan.reshape(4, 4).T, axis=1) for SecondPlan in SampleOfSecondPlan])
    SampleOfSecondArchetypeMargin = np.array([np.sum(SecondPlan.reshape(4, 4), axis=1) for SecondPlan in SampleOfSecondPlan])

    SampleofFirstBarycenterPenalty = np.array([np.linalg.norm(SampleOfFirstBarycenterMargin[i] - SampleOfBarycenter[i]) for i in range(len(Data))])
    SampleofSecondBarycenterPenalty = np.array([np.linalg.norm(SampleOfSecondBarycenterMargin[i] - SampleOfBarycenter[i]) for i in range(len(Data))])

    SampleOfFirstArchetypePenalty = np.array( [np.linalg.norm(SampleOfFirstArchetypeMargin[i] - Archetypes[0]) for i in range(len(Data))])
    SampleOfSecondArchetypePenalty = np.array( [np.linalg.norm(SampleOfSecondArchetypeMargin[i] - Archetypes[1]) for i in range(len(Data))])

    BarycenterPenalty = SampleofFirstBarycenterPenalty + SampleofSecondBarycenterPenalty
    ArchetypePenalty = SampleOfFirstArchetypePenalty + SampleOfSecondArchetypePenalty

    ReluData = np.sum(np.minimum(Data - 0.01, 0), axis=1)
    TotalCost =TransportationCost + 5*ArchetypePenalty + 5*BarycenterPenalty-10*ReluData

    return TotalCost

def ObjectiveExpTransform(Data,PartitionLength,Archetypes):
    NumberOfArchetypes             =len(Archetypes)
    NumberOfAtoms                  = PartitionLength ** 2
    PlanSize                       = NumberOfAtoms ** 2


    T                               = np.linspace(0, 1, PartitionLength)
    couples                         = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x                               = np.array(list(itertools.product(couples, repeat=2)))
    a                               = x[:, 0]
    b                               = x[:, 1]
    CostMatrix                      = np.linalg.norm(a - b, axis=1) ** 2
    CostMatrix                      = CostMatrix+np.zeros((len(Data),1))
    Data                            = np.exp(Data)
    SampleOfBarycenter              = np.array([[Datum[:NumberOfAtoms]] for Datum in Data])
    SampleOfPlanCollection          = np.array([Datum[NumberOfAtoms:].reshape(NumberOfArchetypes, NumberOfAtoms, NumberOfAtoms) for Datum in Data])
    SampleOfSumCollection           = np.array([np.sum(PlanCollection, axis=0).reshape(PlanSize) for PlanCollection in SampleOfPlanCollection])
    TransportationCost              = np.sum(CostMatrix*SampleOfSumCollection,axis=1)
    SampleOfBaryMarginCollection    = np.array([np.sum(PlanCollection, axis=2) for PlanCollection in SampleOfPlanCollection])
    SampleofArchMarginCollection    = np.array([np.sum(PlanCollection.transpose(0, 2, 1), axis=2) for PlanCollection in SampleOfPlanCollection])

    BarycenterPenalty               = np.sum(np.linalg.norm(SampleOfBaryMarginCollection - SampleOfBarycenter, axis=2), axis=1)
    ArchetypePenalty                = np.sum(np.linalg.norm(SampleofArchMarginCollection - Archetypes, axis=2), axis=1)
    TotalCost                       = TransportationCost + 10*ArchetypePenalty + 10*BarycenterPenalty
    return TotalCost

def Objectiveles(Data,PartitionLength,Archetypes):
    NumberOfAtoms                    = PartitionLength ** 2
    PlanSize                         = NumberOfAtoms ** 2


    T                                = np.linspace(0, 1, PartitionLength)
    couples                          = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x                                = np.array(list(itertools.product(couples, repeat=2)))
    a                                = x[:, 0]
    b                                = x[:, 1]
    CostMatrix                       = np.linalg.norm(a - b, axis=1) ** 2

    CostMatrix                       = CostMatrix + np.zeros((len(Data), 1))

    Data                             = np.exp(Data)
    SampleOfBarycenter               = np.array([Datum[0:4]   for Datum in Data])
    SampleOfBarycenter               = np.array([Barycenter/np.sum(Barycenter)   for Barycenter in SampleOfBarycenter])
    SampleOfFirstPlan                = np.array([Datum[4:20]  for Datum in Data])

    SampleOfFirstPlan                = np.array([FirstPlan/np.sum(FirstPlan) for FirstPlan in SampleOfFirstPlan])
    SampleOfSecondPlan               = np.array([Datum[20:36] for Datum in Data])
    SampleOfSecondPlan               = np.array([SecondPlan / np.sum(SecondPlan) for SecondPlan in SampleOfSecondPlan])

    SampleOfFirstTransportationCost  = np.sum(CostMatrix * SampleOfFirstPlan , axis=1)
    SampleOfSecondTransportationCost = np.sum(CostMatrix * SampleOfSecondPlan , axis=1)
    TransportationCost               = SampleOfFirstTransportationCost +SampleOfSecondTransportationCost


    SampleOfFirstBarycenterMargin    = np.array([np.sum(FirstPlan.reshape(4,4).T,     axis=1) for FirstPlan  in SampleOfFirstPlan])
    SampleOfFirstArchetypeMargin     = np.array([np.sum(FirstPlan.reshape(4, 4),      axis=1) for FirstPlan in SampleOfFirstPlan])
    SampleOfSecondBarycenterMargin   = np.array([np.sum(SecondPlan.reshape(4,4).T,    axis=1) for SecondPlan in SampleOfSecondPlan])
    SampleOfSecondArchetypeMargin    = np.array([np.sum(SecondPlan.reshape(4, 4),     axis=1) for SecondPlan in SampleOfSecondPlan])


    SampleofFirstBarycenterPenalty   = np.array([np.linalg.norm(SampleOfFirstBarycenterMargin[i] - SampleOfBarycenter[i]) for i in range(len(Data))])
    SampleofSecondBarycenterPenalty  = np.array([np.linalg.norm(SampleOfSecondBarycenterMargin[i] - SampleOfBarycenter[i]) for i in range(len(Data))])

    SampleOfFirstArchetypePenalty    = np.array([np.linalg.norm(SampleOfFirstArchetypeMargin[i] -Archetypes[0]) for i in range(len(Data))])
    SampleOfSecondArchetypePenalty   = np.array([np.linalg.norm(SampleOfSecondArchetypeMargin[i] -Archetypes[1]) for i in range(len(Data))])


    BarycenterPenalty                = SampleofFirstBarycenterPenalty + SampleofSecondBarycenterPenalty
    ArchetypePenalty                 = SampleOfFirstArchetypePenalty +SampleOfSecondArchetypePenalty

    TotalCost                        =  TransportationCost + 10*ArchetypePenalty + 10*BarycenterPenalty
    return TotalCost


def Objectiveles2(Data,PartitionLength,Archetypes):
    NumberOfAtoms                    = PartitionLength ** 2
    PlanSize                         = NumberOfAtoms ** 2


    T                                = np.linspace(0, 1, PartitionLength)
    couples                          = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x                                = np.array(list(itertools.product(couples, repeat=2)))
    a                                = x[:, 0]
    b                                = x[:, 1]
    CostMatrix                       = np.linalg.norm(a - b, axis=1) ** 2

    CostMatrix                       = CostMatrix + np.zeros((len(Data), 1))

    Data                             = np.exp(Data)
    SampleOfBarycenter               = np.array([Datum[0:4]   for Datum in Data])
    SampleOfBarycenter               = np.array([Barycenter/np.sum(Barycenter)   for Barycenter in SampleOfBarycenter])
    SampleOfFirstPlan                = np.array([Datum[4:20]  for Datum in Data])

    SampleOfFirstPlan                = np.array([FirstPlan/np.sum(FirstPlan) for FirstPlan in SampleOfFirstPlan])
    SampleOfSecondPlan               = np.array([Datum[20:36] for Datum in Data])
    SampleOfSecondPlan               = np.array([SecondPlan / np.sum(SecondPlan) for SecondPlan in SampleOfSecondPlan])

    SampleOfFirstTransportationCost  = np.sum(CostMatrix * SampleOfFirstPlan , axis=1)
    SampleOfSecondTransportationCost = np.sum(CostMatrix * SampleOfSecondPlan , axis=1)
    TransportationCost               = SampleOfFirstTransportationCost +SampleOfSecondTransportationCost


    SampleOfFirstBarycenterMargin    = np.array([np.sum(FirstPlan.reshape(4,4).T,     axis=1) for FirstPlan  in SampleOfFirstPlan])
    SampleOfFirstArchetypeMargin     = np.array([np.sum(FirstPlan.reshape(4, 4),      axis=1) for FirstPlan in SampleOfFirstPlan])
    SampleOfSecondBarycenterMargin   = np.array([np.sum(SecondPlan.reshape(4,4).T,    axis=1) for SecondPlan in SampleOfSecondPlan])
    SampleOfSecondArchetypeMargin    = np.array([np.sum(SecondPlan.reshape(4, 4),     axis=1) for SecondPlan in SampleOfSecondPlan])


    SampleofFirstBarycenterPenalty   = np.array([np.linalg.norm(SampleOfFirstBarycenterMargin[i] - SampleOfBarycenter[i]) for i in range(len(Data))])
    SampleofSecondBarycenterPenalty  = np.array([np.linalg.norm(SampleOfSecondBarycenterMargin[i] - SampleOfBarycenter[i]) for i in range(len(Data))])

    SampleOfFirstArchetypePenalty    = np.array([np.linalg.norm(SampleOfFirstArchetypeMargin[i] -Archetypes[0]) for i in range(len(Data))])
    SampleOfSecondArchetypePenalty   = np.array([np.linalg.norm(SampleOfSecondArchetypeMargin[i] -Archetypes[1]) for i in range(len(Data))])


    BarycenterPenalty                = SampleofFirstBarycenterPenalty + SampleofSecondBarycenterPenalty
    ArchetypePenalty                 = SampleOfFirstArchetypePenalty +SampleOfSecondArchetypePenalty

    TotalCost                        =  TransportationCost + 10*ArchetypePenalty + 10*BarycenterPenalty
    return TotalCost



