import cupy as cp

def Transformation(Sample, Inputtype, Transformationfunction):
    Batchdims = cp.ndim(Sample) - 1
    if Inputtype == "Barycenter" and Transformationfunction == "Sigmoid":
        Sample = cp.exp(Sample)
        BatchNorm = cp.sum(Sample, axis=Batchdims, keepdims=True)
        Sample = Sample / BatchNorm
    elif Inputtype == "Plans" and Transformationfunction == "Sigmoid":
        Sample = cp.exp(Sample)
        BatchNorm = cp.sum(Sample, axis=Batchdims, keepdims=True)
        Sample = Sample / BatchNorm
    elif Inputtype=="Barycenter" and Transformationfunction=="Exponential":
        Sample=cp.exp(Sample)
    elif Inputtype=="Plans" and Transformationfunction=="Exponential":
        Sample=cp.exp(Sample)
    elif Inputtype=="Barycenter" and Transformationfunction=="Normalized":
        BatchNorm=cp.sum(Sample, axis=Batchdims, keepdims=True)
        Sample= Sample / BatchNorm
    elif Inputtype=="Plans" and Transformationfunction=="Normalized":
        BatchNorm=cp.sum(Sample, axis=Batchdims, keepdims=True)
        Sample= Sample / BatchNorm
    return Sample

def Penalty(FirstConfiguration,SecondConfiguration,Metric):
    Batchdims = cp.ndim(FirstConfiguration)
    if Metric=="Square":
        Penalty = cp.sum(cp.linalg.norm(FirstConfiguration - SecondConfiguration, axis=Batchdims - 1), axis=Batchdims - 2)
    if Metric=="Entropy":
        Penalty = cp.sum(cp.sum(cp.abs(FirstConfiguration) * (cp.log(cp.abs(FirstConfiguration)) - cp.log(cp.abs(SecondConfiguration))), axis=Batchdims - 1), axis=Batchdims - 2)
    if Metric == "RevEntropy":
        Penalty = cp.sum(cp.sum(cp.abs(SecondConfiguration) * (cp.log(cp.abs(SecondConfiguration)) - cp.log(cp.abs(FirstConfiguration))), axis=Batchdims - 1), axis=Batchdims - 2)

    return Penalty