import numpy as np





def greedy(loads):
    '''
    propotionally allocate rb resource to slices accordling to their traffic load
    '''
    total_load = sum(loads) +  1e-6
    loads = [load / total_load for load in loads]
    
    return np.array(loads)