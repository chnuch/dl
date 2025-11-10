# Importing Python library 
import numpy as np 

# Define Unit Step Function 
def unitStep(v):  
    if v >= 0: 
        return 1 
    else: 
        return 0 

# Design Perceptron Model 
def perceptronModel(x, w, b): 
    v = np.dot(w, x) + b 
    y = unitStep(v) 
    return y 


# ----------------- OR Gate -----------------
# w1 = 1, w2 = 1, b = -0.5 
def OR_logicFunction(x):  
    w = np.array([1, 1]) 
    b = -0.5 
    return perceptronModel(x, w, b) 


# ----------------- AND Gate -----------------
# w1 = 1, w2 = 1, b = -1.5 
def AND_logicFunction(x):  
    w = np.array([1, 1]) 
    b = -1.5 
    return perceptronModel(x, w, b) 


# ----------------- NOT Gate -----------------
# Single input, w = -1, b = 0.5
def NOT_logicFunction(x):  
    w = np.array([-1]) 
    b = 0.5 
    return perceptronModel(x, w, b) 


# ----------------- XOR Gate (via combination of perceptrons) -----------------
# XOR = (A AND (NOT B)) OR ((NOT A) AND B)
def XOR_logicFunction(x):
    A = x[0]
    B = x[1]
    # Using previously defined perceptrons
    A_AND_NOTB = AND_logicFunction(np.array([A, NOT_logicFunction(np.array([B]))]))
    NOTA_AND_B = AND_logicFunction(np.array([NOT_logicFunction(np.array([A])), B]))
    y = OR_logicFunction(np.array([A_AND_NOTB, NOTA_AND_B]))
    return y


# ----------------- Testing -----------------

# Test inputs
test_inputs = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

print("----- OR Gate -----")
for x in test_inputs:
    print(f"OR({x[0]}, {x[1]}) = {OR_logicFunction(x)}")

print("\n----- AND Gate -----")
for x in test_inputs:
    print(f"AND({x[0]}, {x[1]}) = {AND_logicFunction(x)}")

print("\n----- NOT Gate -----")
for x in [np.array([0]), np.array([1])]:
    print(f"NOT({x[0]}) = {NOT_logicFunction(x)}")

print("\n----- XOR Gate -----")
for x in test_inputs:
    print(f"XOR({x[0]}, {x[1]}) = {XOR_logicFunction(x)}")
