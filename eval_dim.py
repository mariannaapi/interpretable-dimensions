#####33
# Evaluating interpretable dimensions/axes
# Katrin Erk Fall 2023

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

# extended pairwise order consistency:
# pairwise order consistency among test items
# plus order consistency of each test item with respect to each training item
def pairwise_order_consistency_wrt(goldvalues, modelvalues, test_indices):
    if len(goldvalues) != len(modelvalues):
        raise Exception("shouldn't be here")
        
    outcomes = [ ]

    # comparisons among test indices: normal oc_p
    for i1, i2 in itertools.combinations(test_indices, 2):
        goldrel = (goldvalues[i1] > goldvalues[i2])
        modelrel = (modelvalues[i1] > modelvalues[i2])
        outcomes.append(int(goldrel == modelrel))


    # comparison of test to training indices
    for i1 in test_indices:
        for i2 in range(len(goldvalues)):
            if i2 in test_indices: continue
            
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
