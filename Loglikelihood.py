import cupy as cp
import itertools
from TransformationsAndPenalties import *


class Loglikelihood1:

    def __init__(self, args):
        self.args=args

    def Loglikelihoodsimple(self,Batch):
        Archetypes=self.args[0]
        PartitionLength= cp.int(cp.sqrt(len(Archetypes[0])))
        NumberOfAtoms = PartitionLength ** 2
        NumberOfArchetypes=len(Archetypes)

        #Creating the cost Matrix
        T = cp.linspace(0, 1, PartitionLength)
        couples = cp.array(cp.meshgrid(T, T)).T.reshape(-1, 2)
        x = cp.array(list(itertools.product(couples, repeat=2)))
        a = x[:, 0]
        b = x[:, 1]

        CostMatrix = cp.linalg.norm(a - b, axis=1) ** 2
        CostMatrix = CostMatrix + cp.zeros((len(Batch), NumberOfArchetypes, 1))
        Archetypes = Archetypes + cp.zeros((len(Batch), NumberOfArchetypes, 1))



        #Isolating the Barycenter from the rest of the input
        Barycenter         = Batch[:,:NumberOfAtoms]

        # We apply the tranformation on the Barycenter.

        Barycenter         = Transformation(Barycenter,"Barycenter",self.args[1])

        # Creating a copy of the Barycenter candidate that will correspond to each Archetype

        Barycenter         = cp.repeat(cp.array([Barycenter]), NumberOfArchetypes, axis=0).transpose(1, 0, 2)

        #Isolating the Plans from the rest of the input

        Plans              = Batch[:,NumberOfAtoms:]

        #Spliting The plans
        Plans              = cp.array(cp.split(Plans, NumberOfArchetypes, axis=1)).transpose(1, 0, 2)

        #We apply the tranformation on the plans
        Plans              = Transformation(Plans,"Plans",self.args[1])

        #Calculating the transportaion cost
        TransportationCost = cp.sum(cp.sum(CostMatrix[:, :, :] * Plans[:, :, :], axis=2), axis=1)

        #Calculating the Margins of each plan.
        BarycenterMargin   = cp.sum(Plans.reshape(-1, NumberOfArchetypes, NumberOfAtoms, NumberOfAtoms).transpose(0, 1, 3, 2), axis=3)
        ArchetypeMargin   = cp.sum(Plans.reshape(-1, NumberOfArchetypes, NumberOfAtoms, NumberOfAtoms), axis=3)

        #Calculating the penalty for each plan.
        BarycenterPenalty = Penalty(BarycenterMargin,Barycenter,self.args[2])
        ArchetypePenalty  = Penalty(ArchetypeMargin,Archetypes,self.args[3])

        # This punishes negative values.

        if self.args[4] == "Yes":
            ReluData = -cp.sum(cp.minimum(Batch, cp.zeros((len(Batch), len(Batch[0])))), axis=1)

        else :
            ReluData=0

        TotalCost = TransportationCost + 10*ArchetypePenalty + 10*BarycenterPenalty + 10*ReluData

        return TotalCost


class Loglikelihood2:

    def __init__(self, args):
        self.args=args

    def Loglikelihoodsimple(self,Sample):

        Archetypes = self.args[0]

        PartitionLength = cp.int(cp.sqrt(len(Archetypes[0])))
        NumberOfAtoms = PartitionLength ** 2
        NumberOfArchetypes = len(Archetypes)

        # Creating the cost Matrix
        T = cp.linspace(0, 1, PartitionLength)
        couples = cp.array(cp.meshgrid(T, T)).T.reshape(-1, 2)
        x = cp.array(list(itertools.product(couples, repeat=2)))
        a = x[:, 0]
        b = x[:, 1]

        CostMatrix = cp.linalg.norm(a - b, axis=1) ** 2


        # Isolating the Barycenter from the rest of the input
        Barycenter = Sample[ :NumberOfAtoms]

        # We apply the tranformation on the Barycenter.
        Barycenter = Transformation(Barycenter, "Barycenter", self.args[1])

        # Creating a copy of the Barycenter candidate that will correspond to each Archetype

        Barycenter = cp.repeat(cp.array([Barycenter]), NumberOfArchetypes, axis=0)


        # Isolating the Plans from the rest of the input
        Plans = Sample[ NumberOfAtoms:]

        # Spliting The plans
        Plans = cp.array(cp.split(Plans, NumberOfArchetypes, axis=0))

        # We apply the tranformation on the plans
        Plans = Transformation(Plans, "Plans", self.args[1])

        # Calculating the transportaion cost
        TransportationCost = cp.sum(cp.sum(Plans * CostMatrix, axis=1), axis=0)


        # Calculating the Margins of each plan.
        BarycenterMargin = cp.sum(Plans.reshape(NumberOfArchetypes, NumberOfAtoms, NumberOfAtoms).transpose(0, 2, 1),
                                  axis=2)
        ArchetypeMargin = cp.sum(Plans.reshape(NumberOfArchetypes, NumberOfAtoms, NumberOfAtoms), axis=2)



        # Calculating the penalty for each plan.

        BarycenterPenalty  =  Penalty(BarycenterMargin, Barycenter, self.args[2])
        ArchetypePenalty   =  Penalty(ArchetypeMargin, Archetypes, self.args[3])


        # This punishes negative values.

        if self.args[4] == "Yes":
            ReluData = - cp.sum(cp.minimum(Sample, cp.zeros(len(Sample))), axis=0)

        else:
            ReluData = 0


        TotalCost = TransportationCost + 10 * ArchetypePenalty + 10 * BarycenterPenalty + 10 * ReluData

        return TotalCost


