import numpy as np

def step_function(x):
    return np.where(x > 0, 1, 0).astype(np.int64)

#print (step_function(3.0))
#print (step_function(-3.0))
#print (step_function(np.arange(3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0)) )   


y = step_function(np.array([1.0, 2.0, -3.0]))
#sy = y.astype(np.int)
print (y)
