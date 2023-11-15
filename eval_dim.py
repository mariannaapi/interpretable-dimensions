import itertools

# pairwise order consistency, normal definition
def pairwise_order_consistency(goldvalues, modelvalues):
    if len(goldvalues) != len(modelvalues):
        raise Exception("shouldn't be here")
        
    outcomes = [ ]
    for i1, i2 in itertools.combinations(range(len(goldvalues)), 2):
        goldrel = (goldvalues[i1] > goldvalues[i2])
        modelrel = (modelvalues[i1] > modelvalues[i2])
        outcomes.append(int(goldrel == modelrel))

    if len(outcomes) == 0:
        return None
    
    return sum(outcomes) / len(outcomes)

# pairwise order consistency only for a subset of the model values 
def pairwise_order_consistency_wrt(goldvalues, modelvalues, test_indices):
    if len(goldvalues) != len(modelvalues):
        raise Exception("shouldn't be here")
        
    outcomes = [ ]
    for i1 in test_indices:
        for i2 in range(len(goldvalues)):
            if i1 == i2: continue
            
            goldrel = (goldvalues[i1] > goldvalues[i2])
            modelrel = (modelvalues[i1] > modelvalues[i2])
            outcomes.append(int(goldrel == modelrel))
        
    if len(outcomes) == 0:
        return None

    return sum(outcomes) / len(outcomes)

# mean squared error
def mean_squared_error(goldvalues, modelvalues):
    if len(goldvalues) != len(modelvalues):
        raise Exception("shouldn't be here")

    if len(goldvalues) == 0:
        return None
    
    return sum([(g - m)**2 for g, m in zip(goldvalues, modelvalues)]) / len(goldvalues)
